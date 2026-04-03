"""Execution orchestrator: coordinates orders, risk, and positions."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.execution.ibkr import IBKRConnector, Position
    from src.execution.orders import OrderManager
    from src.execution.risk import RiskManager
    from src.models.core import TradeProposal

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class ExecutionResult:
    """Result of trade execution attempt."""

    success: bool
    """Whether execution succeeded."""

    ticker: str
    order_id: str | None = None
    """Order ID if placed."""

    shares: int | None = None
    """Shares in position."""

    entry_price: float | None = None
    """Entry price achieved."""

    error: str | None = None
    """Error message if failed."""

    risk_checks: list[str] | None = None
    """Risk check failures if validation failed."""


@dataclass(slots=True)
class PositionMonitorState:
    """Tracking state for time-based stagnation exits."""

    opened_at: datetime
    anchor_price: float


class TradeExecutor:
    """Coordinates trade execution across risk, orders, and broker."""

    def __init__(
        self,
        ibkr: IBKRConnector,
        order_manager: OrderManager,
        risk_manager: RiskManager,
    ) -> None:
        """Initialize executor.

        Args:
            ibkr: IBKR connector
            order_manager: Order manager handling bracket orders
            risk_manager: Risk manager enforcing limits
        """
        self.ibkr = ibkr
        self.order_manager = order_manager
        self.risk_manager = risk_manager
        self.execution_log: list[ExecutionResult] = []
        self._position_monitor_state: dict[str, PositionMonitorState] = {}
        """Log of all execution results."""

    async def execute_proposal(
        self,
        proposal: TradeProposal,
        current_prices: dict[str, float],
    ) -> ExecutionResult:
        """Execute a trade proposal with full risk and order management.

        Args:
            proposal: Trade proposal to execute
            current_prices: Dict of ticker → current market price

        Returns:
            ExecutionResult with success status and details
        """
        current_price = current_prices.get(proposal.ticker, 0.0)
        if current_price <= 0:
            current_price = await self.ibkr.get_quote(proposal.ticker)

        # Step 1: Validate proposal structure
        is_valid, validation_errors = await self.order_manager.validate_proposal_for_execution(
            proposal, current_price
        )
        if not is_valid:
            result = ExecutionResult(
                success=False,
                ticker=proposal.ticker,
                error="Proposal validation failed",
                risk_checks=validation_errors,
            )
            self.execution_log.append(result)
            logger.warning(
                "proposal_validation_failed",
                ticker=proposal.ticker,
                errors=validation_errors,
            )
            return result

        # Step 2: Get risk checks
        account_value = await self.ibkr.get_account_value()
        is_approved, risk_reasons = await self.risk_manager.validate_trade_for_risk(
            proposal, current_price, account_value
        )

        if not is_approved:
            result = ExecutionResult(
                success=False,
                ticker=proposal.ticker,
                error="Risk validation failed",
                risk_checks=risk_reasons,
            )
            self.execution_log.append(result)
            logger.warning(
                "risk_validation_failed",
                ticker=proposal.ticker,
                reasons=risk_reasons,
            )
            return result

        # Step 3: Calculate position size
        sizing = await self.risk_manager.calculate_position_size(
            proposal, current_price, account_value
        )

        if not sizing:
            result = ExecutionResult(
                success=False,
                ticker=proposal.ticker,
                error="Position sizing failed",
            )
            self.execution_log.append(result)
            logger.error("position_sizing_failed", ticker=proposal.ticker)
            return result

        # Step 4: Place bracket order
        try:
            order_id = await self.order_manager.place_bracket_order(
                ticker=proposal.ticker,
                side="BUY" if proposal.direction == "long" else "SELL",
                quantity=sizing.shares,
                entry_limit_price=None,
                stop_loss_pct=proposal.stop_loss_pct,
                take_profit_pct=proposal.take_profit_pct,
            )

            result = ExecutionResult(
                success=True,
                ticker=proposal.ticker,
                order_id=order_id,
                shares=sizing.shares,
                entry_price=current_price,
            )

            self.execution_log.append(result)
            self.risk_manager.register_trade(proposal.ticker)

            logger.info(
                "trade_executed",
                order_id=order_id,
                ticker=proposal.ticker,
                direction=proposal.direction,
                shares=sizing.shares,
                entry_price=current_price,
                stop_loss_pct=proposal.stop_loss_pct,
                take_profit_pct=proposal.take_profit_pct,
            )

            return result

        except Exception as e:
            error_msg = f"Order placement failed: {str(e)}"
            result = ExecutionResult(
                success=False,
                ticker=proposal.ticker,
                error=error_msg,
            )
            self.execution_log.append(result)
            logger.error("order_placement_error", ticker=proposal.ticker, error=str(e))
            return result

    async def monitor_and_update_positions(self) -> None:
        """Monitor open positions and bracket orders for fills/stops."""
        # Monitor bracket order exits
        await self.order_manager.monitor_brackets()

        # Get updated positions
        positions = await self.ibkr.get_positions()

        # Update market prices for all open positions
        for ticker, position in positions.items():
            current_price = await self.ibkr.get_quote(ticker)
            if current_price:
                position.update_market_price(current_price)

        await self._apply_time_based_exits(positions)

        # Log current portfolio state
        logger.info(
            "positions_updated",
            position_count=len(positions),
            bracket_count=len(self.order_manager.get_active_brackets()),
        )

    async def close_position(
        self,
        ticker: str,
        reason: str = "manual",
    ) -> ExecutionResult:
        """Manually close an open position.

        Args:
            ticker: Ticker to close
            reason: Reason for close (e.g., "manual", "stop_loss", "take_profit")

        Returns:
            ExecutionResult of close order
        """
        positions = await self.ibkr.get_positions()
        position = positions.get(ticker)

        if not position or position.quantity == 0:
            result = ExecutionResult(
                success=False,
                ticker=ticker,
                error=f"No open position for {ticker}",
            )
            self.execution_log.append(result)
            logger.warning("position_close_failed_no_position", ticker=ticker)
            return result

        side = "SELL" if position.quantity > 0 else "BUY"
        quantity = abs(position.quantity)

        # Use market order for time-based stagnation exits.
        use_market_order = reason == "time_exit_stagnation"

        current_price = await self.ibkr.get_quote(ticker)
        limit_price = None
        if not use_market_order and current_price:
            if side == "SELL":
                limit_price = current_price * 0.99
            else:
                limit_price = current_price * 1.01

        try:
            close_order_id = await self.order_manager.close_position(
                ticker=ticker,
                quantity=quantity,
                side=side,
                limit_price=limit_price,
                market_order=use_market_order,
            )

            result = ExecutionResult(
                success=True,
                ticker=ticker,
                order_id=close_order_id,
                shares=quantity,
                entry_price=position.entry_price,
            )

            self.execution_log.append(result)

            logger.info(
                "position_closed",
                ticker=ticker,
                order_id=close_order_id,
                quantity=quantity,
                side=side,
                reason=reason,
            )

            return result

        except Exception as e:
            result = ExecutionResult(
                success=False,
                ticker=ticker,
                error=f"Close order failed: {str(e)}",
            )
            self.execution_log.append(result)
            logger.error("position_close_error", ticker=ticker, error=str(e))
            return result

    async def _apply_time_based_exits(self, positions: dict[str, Position]) -> None:
        """Close stagnant positions after 3 days if price stayed within +/-1%."""
        now = datetime.now(timezone.utc)

        # Drop monitor state for symbols that are no longer open.
        open_symbols = set(positions.keys())
        for symbol in list(self._position_monitor_state.keys()):
            if symbol not in open_symbols:
                del self._position_monitor_state[symbol]

        for ticker, position in positions.items():
            current_price = position.current_price if position.current_price > 0 else await self.ibkr.get_quote(ticker)
            if current_price <= 0:
                continue

            state = self._position_monitor_state.get(ticker)
            if state is None:
                self._position_monitor_state[ticker] = PositionMonitorState(
                    opened_at=now,
                    anchor_price=current_price,
                )
                continue

            age = now - state.opened_at
            move_pct = abs((current_price - state.anchor_price) / state.anchor_price) * 100.0
            if age >= timedelta(days=3) and move_pct <= 1.0:
                logger.info(
                    "time_based_exit_triggered",
                    ticker=ticker,
                    age_hours=round(age.total_seconds() / 3600.0, 2),
                    move_pct=round(move_pct, 3),
                )
                await self.close_position(ticker=ticker, reason="time_exit_stagnation")

    async def get_portfolio_summary(self) -> dict:
        """Get summary of current portfolio state.

        Returns:
            Dict with positions, cash, P&L, active brackets
        """
        account_value = await self.ibkr.get_account_value()
        cash = await self.ibkr.get_cash_balance()
        positions = await self.ibkr.get_positions()

        total_market_value = account_value - cash
        active_brackets = self.order_manager.get_active_brackets()

        return {
            "account_value": account_value,
            "cash": cash,
            "invested": total_market_value,
            "position_count": len(positions),
            "active_brackets": len(active_brackets),
            "bracket_statuses": self.order_manager.get_bracket_summary(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_execution_log(self, limit: int | None = None) -> list[ExecutionResult]:
        """Get execution log.

        Args:
            limit: Max number of recent results to return

        Returns:
            List of ExecutionResult objects
        """
        if limit:
            return self.execution_log[-limit:]
        return self.execution_log
