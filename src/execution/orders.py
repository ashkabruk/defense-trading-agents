"""Order management with bracket orders and position monitoring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.execution.ibkr import IBKRConnector
    from src.models.core import TradeProposal

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class BracketOrder:
    """Bracket order: entry + stop-loss + take-profit."""

    entry_order_id: str
    stop_loss_order_id: str | None = None
    take_profit_order_id: str | None = None
    trailing_stop_order_id: str | None = None
    ticker: str | None = None
    side: str = "BUY"
    entry_quantity: int = 0
    entry_price: float | None = None
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    status: str = "pending"  # pending | entry_filled | exit_filled | cancelled
    entry_fill_price: float | None = None
    trailing_activated: bool = False


class OrderManager:
    """Manages bracket construction and exit leg monitoring."""

    def __init__(self, ibkr: IBKRConnector) -> None:
        self.ibkr = ibkr
        self.brackets: dict[str, BracketOrder] = {}

    async def place_bracket_order(
        self,
        ticker: str,
        side: str,
        quantity: int,
        entry_limit_price: float | None,
        stop_loss_pct: float,
        take_profit_pct: float,
    ) -> str:
        """Place entry order and register bracket with computed stop/profit levels."""
        side = side.upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        if quantity <= 0:
            raise ValueError("quantity must be > 0")

        entry_price = entry_limit_price
        if entry_price is None or entry_price <= 0:
            # Required behavior: use last price because no real-time bid/ask subscriptions.
            entry_price = await self.ibkr.get_quote(ticker)

        if side == "BUY":
            stop_loss_price = entry_price * (1 - stop_loss_pct / 100.0)
            take_profit_price = entry_price * (1 + take_profit_pct / 100.0)
        else:
            stop_loss_price = entry_price * (1 + stop_loss_pct / 100.0)
            take_profit_price = entry_price * (1 - take_profit_pct / 100.0)

        entry_order_id = await self.ibkr.place_order(
            ticker=ticker,
            side=side,
            quantity=quantity,
            limit_price=entry_price,
        )

        self.brackets[entry_order_id] = BracketOrder(
            entry_order_id=entry_order_id,
            ticker=ticker,
            side=side,
            entry_quantity=quantity,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
        )

        logger.info(
            "bracket_order_created",
            entry_order_id=entry_order_id,
            ticker=ticker,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
        )
        return entry_order_id

    async def monitor_brackets(self) -> None:
        """Monitor bracket orders, place/cancel exit legs on fills, and handle trailing stops."""
        orders = await self.ibkr.get_orders()

        for entry_order_id, bracket in list(self.brackets.items()):
            entry_order = orders.get(entry_order_id)
            if entry_order is None:
                continue

            if entry_order.status.value == "filled" and bracket.status == "pending":
                bracket.status = "entry_filled"
                bracket.entry_fill_price = entry_order.filled_price

                exit_side = "SELL" if bracket.side == "BUY" else "BUY"

                if bracket.stop_loss_price is not None:
                    bracket.stop_loss_order_id = await self.ibkr.place_order(
                        ticker=bracket.ticker or "",
                        side=exit_side,
                        quantity=bracket.entry_quantity,
                        stop_price=bracket.stop_loss_price,
                    )

                if bracket.take_profit_price is not None:
                    bracket.take_profit_order_id = await self.ibkr.place_order(
                        ticker=bracket.ticker or "",
                        side=exit_side,
                        quantity=bracket.entry_quantity,
                        limit_price=bracket.take_profit_price,
                    )

                logger.info(
                    "bracket_entry_filled",
                    entry_order_id=entry_order_id,
                    fill_price=bracket.entry_fill_price,
                    stop_order_id=bracket.stop_loss_order_id,
                    take_profit_order_id=bracket.take_profit_order_id,
                )

            if bracket.status != "entry_filled":
                continue

            # Apply trailing stop-loss natively once gain exceeds 1%.
            if bracket.entry_fill_price:
                current_price = await self.ibkr.get_quote(bracket.ticker or "")
                gain_pct = (
                    ((current_price - bracket.entry_fill_price) / bracket.entry_fill_price) * 100.0
                    if bracket.entry_fill_price > 0
                    else 0.0
                )

                if (
                    bracket.side == "BUY"
                    and gain_pct > 1.0
                    and not bracket.trailing_activated
                ):
                    await self._activate_trailing_stop(bracket, current_price)

            stop_filled = False
            tp_filled = False
            trailing_filled = False

            if bracket.stop_loss_order_id:
                stop = orders.get(bracket.stop_loss_order_id)
                stop_filled = bool(stop and stop.status.value == "filled")

            if bracket.take_profit_order_id:
                take_profit = orders.get(bracket.take_profit_order_id)
                tp_filled = bool(take_profit and take_profit.status.value == "filled")

            if bracket.trailing_stop_order_id:
                trailing = orders.get(bracket.trailing_stop_order_id)
                trailing_filled = bool(trailing and trailing.status.value == "filled")

            if stop_filled or tp_filled or trailing_filled:
                await self._cancel_bracket_exit_legs(bracket)
                bracket.status = "exit_filled"
                logger.info(
                    "bracket_exit_filled",
                    entry_order_id=entry_order_id,
                    stop_filled=stop_filled,
                    take_profit_filled=tp_filled,
                    trailing_filled=trailing_filled,
                )

    async def _cancel_bracket_exit_legs(self, bracket: BracketOrder) -> None:
        """Cancel any unfilled exit legs in the bracket."""
        if bracket.stop_loss_order_id:
            await self.ibkr.cancel_order(bracket.stop_loss_order_id)
        if bracket.take_profit_order_id:
            await self.ibkr.cancel_order(bracket.take_profit_order_id)
        if bracket.trailing_stop_order_id:
            await self.ibkr.cancel_order(bracket.trailing_stop_order_id)

    async def _activate_trailing_stop(self, bracket: BracketOrder, current_price: float) -> None:
        """Replace fixed stop with native IBKR trailing stop that locks in half current gain."""
        if not bracket.entry_fill_price:
            return

        gain = current_price - bracket.entry_fill_price
        if gain <= 0:
            return

        desired_stop_price = bracket.entry_fill_price + gain * 0.5
        trailing_amount = max(current_price - desired_stop_price, 0.01)

        if bracket.stop_loss_order_id:
            await self.ibkr.cancel_order(bracket.stop_loss_order_id)
            logger.info(
                "trailing_stop_cancelled_fixed_stop",
                order_id=bracket.stop_loss_order_id,
            )

        exit_side = "SELL" if bracket.side == "BUY" else "BUY"
        trailing_order_id = await self.ibkr.place_order(
            ticker=bracket.ticker or "",
            side=exit_side,
            quantity=bracket.entry_quantity,
            stop_price=desired_stop_price,
            trailing_amount=trailing_amount,
        )
        bracket.stop_loss_order_id = None
        bracket.stop_loss_price = desired_stop_price
        bracket.trailing_stop_order_id = trailing_order_id
        bracket.trailing_activated = True

        logger.info(
            "trailing_stop_activated",
            ticker=bracket.ticker,
            trailing_order_id=trailing_order_id,
            desired_stop_price=desired_stop_price,
            trailing_amount=trailing_amount,
        )

    async def close_position(
        self,
        ticker: str,
        quantity: int,
        side: str,
        limit_price: float | None = None,
        market_order: bool = False,
    ) -> str:
        """Close an open position using either a market order or a limit fallback."""
        if not market_order and (limit_price is None or limit_price <= 0):
            limit_price = await self.ibkr.get_quote(ticker)
        close_order_id = await self.ibkr.place_order(
            ticker=ticker,
            side=side,
            quantity=quantity,
            limit_price=limit_price,
            market_order=market_order,
        )
        logger.info(
            "position_close_requested",
            ticker=ticker,
            quantity=quantity,
            side=side,
            market_order=market_order,
            close_order_id=close_order_id,
            limit_price=limit_price,
        )
        return close_order_id

    def get_active_brackets(self) -> list[BracketOrder]:
        """Return all brackets that have not fully exited."""
        return [b for b in self.brackets.values() if b.status not in {"exit_filled", "cancelled"}]

    def get_bracket_summary(self) -> dict[str, int]:
        """Return count of brackets by status."""
        summary: dict[str, int] = {}
        for bracket in self.brackets.values():
            summary[bracket.status] = summary.get(bracket.status, 0) + 1
        return summary

    async def validate_proposal_for_execution(
        self,
        proposal: TradeProposal,
        current_price: float,
    ) -> tuple[bool, list[str]]:
        """Validate proposal fields required for bracket order execution."""
        reasons: list[str] = []

        if proposal.direction not in {"long", "short"}:
            reasons.append(f"Invalid direction: {proposal.direction}")

        if proposal.stop_loss_pct <= 0:
            reasons.append("stop_loss_pct must be > 0")

        if proposal.take_profit_pct <= 0:
            reasons.append("take_profit_pct must be > 0")

        if proposal.position_size_pct <= 0:
            reasons.append("position_size_pct must be > 0")

        if current_price <= 0:
            # Execution can still fetch last price from IB; this is only a warning-level guard.
            reasons.append("Current price unavailable")

        return len(reasons) == 0, reasons
