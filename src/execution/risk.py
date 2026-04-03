"""Risk management enforcing hard limits from risk.yaml."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.execution.ibkr import IBKRConnector, Position
    from src.models.config import RiskConfig
    from src.models.core import TradeProposal

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class PositionSizing:
    """Result of position sizing calculation."""

    symbol: str
    direction: str
    shares: int
    account_pct: float
    risk_amount_usd: float
    risk_pct_of_capital: float
    entry_price: float
    stop_loss_price: float


@dataclass(slots=True)
class RiskCheckResult:
    """Result of risk checking."""

    passed: bool
    reasons: list[str]
    position_size_pct: float | None = None
    available_cash: float | None = None
    max_loss_pct: float | None = None


class RiskManager:
    """Risk enforcer for hard limits from config/risk.yaml."""

    def __init__(self, risk_config: RiskConfig, ibkr: IBKRConnector) -> None:
        self.risk_config = risk_config
        self.ibkr = ibkr

        self.trades_today: list[tuple[str, datetime]] = []
        self.daily_realized_loss_usd: float = 0.0
        self._correlation_matrix = self._build_defense_correlation_matrix()

    async def calculate_position_size(
        self,
        proposal: TradeProposal,
        current_price: float,
        account_value: float,
    ) -> PositionSizing | None:
        """
        Calculate conviction-weighted position size: base_size * (conviction / 7) * materiality_ratio.
        Higher conviction + materiality = larger position, capped at max_position_pct.
        """
        if current_price <= 0 or account_value <= 0:
            return None

        # Calculate conviction multiplier: 0.6->0.43, 7->1.0, 10->1.43
        conviction_multiplier = float(proposal.conviction) / 7.0

        # Load materiality ratio from fundamentals
        materiality_ratio = await self._get_materiality_ratio(proposal.ticker)

        # Calculate conviction-weighted base size
        base_position_pct = float(proposal.position_size_pct)
        conviction_weighted_pct = base_position_pct * conviction_multiplier * materiality_ratio

        # Apply correlation reduction if portfolio has correlated positions
        correlation_adjustment = await self._get_correlation_adjustment(proposal.ticker)
        conviction_weighted_pct *= correlation_adjustment

        # Cap at max_position_pct
        capped_pct = min(conviction_weighted_pct, self.risk_config.max_position_pct)

        target_value = account_value * (capped_pct / 100.0)
        shares = int(target_value / current_price)
        if shares <= 0:
            return None

        stop_loss_pct = float(proposal.stop_loss_pct)
        if proposal.direction == "long":
            stop_loss_price = current_price * (1 - stop_loss_pct / 100.0)
            per_share_risk = current_price - stop_loss_price
        else:
            stop_loss_price = current_price * (1 + stop_loss_pct / 100.0)
            per_share_risk = stop_loss_price - current_price

        risk_amount = max(per_share_risk, 0.0) * shares
        risk_pct = (risk_amount / account_value) * 100.0 if account_value > 0 else 0.0

        return PositionSizing(
            symbol=proposal.ticker,
            direction=proposal.direction,
            shares=shares,
            account_pct=(shares * current_price / account_value) * 100.0,
            risk_amount_usd=risk_amount,
            risk_pct_of_capital=risk_pct,
            entry_price=current_price,
            stop_loss_price=stop_loss_price,
        )

    async def validate_trade_for_risk(
        self,
        proposal: TradeProposal,
        current_price: float,
        account_value: float,
    ) -> tuple[bool, list[str]]:
        """Compatibility helper used by executor."""
        result = await self.check_proposal(proposal, {proposal.ticker: current_price}, account_value)
        return result.passed, result.reasons

    async def check_proposal(
        self,
        proposal: TradeProposal,
        current_prices: dict[str, float],
        account_value: float | None = None,
    ) -> RiskCheckResult:
        """Run full hard-limit risk checks for a proposal."""
        reasons: list[str] = []

        price = current_prices.get(proposal.ticker, 0.0)
        if price <= 0:
            reasons.append(f"No market price available for {proposal.ticker}")
            return RiskCheckResult(False, reasons)

        if account_value is None:
            account_value = await self.ibkr.get_account_value()

        cash = await self.ibkr.get_cash_balance()
        positions = await self.ibkr.get_positions()

        self._cleanup_trades_today()

        if proposal.ticker not in self.risk_config.allowed_tickers:
            reasons.append(f"Ticker {proposal.ticker} not in allowed_tickers")

        if proposal.conviction < self.risk_config.min_conviction_score:
            reasons.append(
                f"Conviction {proposal.conviction:.1f} below minimum {self.risk_config.min_conviction_score}"
            )

        if proposal.ragas_score is not None and proposal.ragas_score < self.risk_config.min_ragas_score:
            reasons.append(
                f"RAGAS {proposal.ragas_score:.2f} below minimum {self.risk_config.min_ragas_score:.2f}"
            )

        if proposal.max_holding_days > self.risk_config.max_holding_days:
            reasons.append(
                f"max_holding_days {proposal.max_holding_days} exceeds {self.risk_config.max_holding_days}"
            )

        if len(positions) >= self.risk_config.max_open_positions:
            reasons.append(
                f"Open positions {len(positions)} exceeds max_open_positions {self.risk_config.max_open_positions}"
            )

        if len(self.trades_today) >= self.risk_config.max_trades_per_day:
            reasons.append(
                f"Trades today {len(self.trades_today)} exceeds max_trades_per_day {self.risk_config.max_trades_per_day}"
            )

        if self._is_ticker_in_cooldown(proposal.ticker):
            reasons.append(
                f"Ticker {proposal.ticker} is in cooldown ({self.risk_config.cool_down_minutes}m)"
            )

        if self._is_ticker_in_earnings_blackout(proposal.ticker):
            reasons.append(
                f"Ticker {proposal.ticker} is within earnings blackout window (5 trading days)"
            )

        if proposal.position_size_pct > self.risk_config.max_position_pct:
            reasons.append(
                f"Position size {proposal.position_size_pct:.2f}% exceeds max_position_pct {self.risk_config.max_position_pct:.2f}%"
            )

        sizing = await self.calculate_position_size(proposal, price, account_value)
        if sizing is None:
            reasons.append("Position size calculation failed")
            max_loss_pct = None
            position_size_pct = None
        else:
            position_size_pct = sizing.account_pct
            max_loss_pct = sizing.risk_pct_of_capital

            if proposal.direction == "long":
                notional_cost = sizing.shares * price
                if notional_cost > cash:
                    reasons.append(
                        f"Cash ${cash:.2f} insufficient for long position cost ${notional_cost:.2f}"
                    )

        if account_value > 0:
            realized_loss_pct = (self.daily_realized_loss_usd / account_value) * 100.0
            if realized_loss_pct > self.risk_config.max_daily_loss_pct:
                reasons.append(
                    f"Daily realized loss {realized_loss_pct:.2f}% exceeds max_daily_loss_pct {self.risk_config.max_daily_loss_pct:.2f}%"
                )

        concentration_pct = self._estimate_sector_concentration_pct(positions, proposal.ticker)
        if concentration_pct > self.risk_config.max_sector_concentration_pct:
            reasons.append(
                f"Sector concentration {concentration_pct:.2f}% exceeds max_sector_concentration_pct {self.risk_config.max_sector_concentration_pct:.2f}%"
            )

        passed = len(reasons) == 0
        logger.info(
            "risk_check_completed",
            passed=passed,
            ticker=proposal.ticker,
            reason_count=len(reasons),
        )

        return RiskCheckResult(
            passed=passed,
            reasons=reasons,
            position_size_pct=position_size_pct,
            available_cash=cash,
            max_loss_pct=max_loss_pct,
        )

    def register_trade(self, ticker: str, was_loss: bool = False) -> None:
        """Register an executed trade for daily limits and cooldown tracking."""
        now = datetime.now(timezone.utc)
        self.trades_today.append((ticker, now))
        logger.info("trade_registered", ticker=ticker, trades_today=len(self.trades_today), was_loss=was_loss)

    def register_realized_pnl(self, pnl_usd: float) -> None:
        """Register closed-trade PnL for daily loss guardrails."""
        if pnl_usd < 0:
            self.daily_realized_loss_usd += abs(pnl_usd)

    def reset_daily_counters(self) -> None:
        """Reset daily counters, typically at start of trading day."""
        self.trades_today.clear()
        self.daily_realized_loss_usd = 0.0

    def _cleanup_trades_today(self) -> None:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=1)
        self.trades_today = [(ticker, ts) for ticker, ts in self.trades_today if ts > cutoff]

    def _is_ticker_in_cooldown(self, ticker: str) -> bool:
        if self.risk_config.cool_down_minutes <= 0:
            return False
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=self.risk_config.cool_down_minutes)
        return any(t == ticker and ts > cutoff for t, ts in self.trades_today)

    def _estimate_sector_concentration_pct(
        self,
        current_positions: dict[str, Position],
        new_ticker: str,
    ) -> float:
        """Simple count-based concentration for the defense-only universe."""
        allowed = set(self.risk_config.allowed_tickers)
        if not allowed:
            return 0.0

        defense_positions = [sym for sym in current_positions.keys() if sym in allowed]
        if new_ticker in allowed and new_ticker not in defense_positions:
            defense_positions.append(new_ticker)

        return (len(defense_positions) / len(allowed)) * 100.0

    def _is_ticker_in_earnings_blackout(self, ticker: str) -> bool:
        """Check if ticker is within 5 trading days before earnings."""
        try:
            with open("src/data/earnings_blackout_calendar.json", "r") as f:
                calendar = json.load(f)
        except Exception as e:
            logger.warning("earnings_calendar_load_error", error=str(e))
            return False
        
        earnings_dates = calendar.get("earnings_dates", [])
        ticker_earnings = next((e for e in earnings_dates if e["ticker"] == ticker), None)
        
        if ticker_earnings is None:
            return False
        
        earnings_date_str = ticker_earnings.get("next_earnings_date")
        try:
            earnings_date = datetime.strptime(earnings_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return False
        
        now = datetime.now(timezone.utc)
        days_until_earnings = (earnings_date - now).days
        
        # Count trading days (Mon-Fri) until earnings
        trading_days_count = 0
        current_date = now.date()
        
        while current_date < earnings_date.date() and trading_days_count < 5:
            weekday = current_date.weekday()  # 0=Monday, 4=Friday
            if weekday < 5:  # Monday through Friday
                trading_days_count += 1
            current_date += timedelta(days=1)
        
        # If earnings within 5 trading days, return True
        return trading_days_count < 5 and days_until_earnings >= 0

    async def _get_materiality_ratio(self, ticker: str) -> float:
        """
        Get materiality ratio for a ticker (contract value / annual revenue).
        Returns between 0.5 (low materiality) and 1.5 (high materiality).
        """
        try:
            with open("src/data/company_fundamentals.json", "r") as f:
                data = json.load(f)
                fundamentals_map = {f["ticker"]: f for f in data.get("fundamentals", [])}
        except Exception as e:
            logger.warning("materiality_ratio_load_error", ticker=ticker, error=str(e))
            return 1.0  # Default neutral ratio
        
        ticker_upper = ticker.upper()
        if ticker_upper not in fundamentals_map:
            return 1.0
        
        # Use defense_revenue_pct as proxy for materiality
        # Higher defense revenue % = contracts more material to stock price
        defense_pct = fundamentals_map[ticker_upper].get("defense_revenue_pct", 0.5)
        # Scale from 0.5x to 1.5x based on 20% (low) to 100% defense revenue
        materiality_multiplier = 0.5 + (defense_pct * 1.0)
        return min(1.5, max(0.5, materiality_multiplier))

    async def _get_correlation_adjustment(self, new_ticker: str) -> float:
        """
        Reduce position size if portfolio already has correlated defense tickers.
        Uses a hardcoded pairwise correlation matrix for the defense universe.
        """
        positions = await self.ibkr.get_positions()
        if not positions:
            return 1.0

        ticker = new_ticker.upper()
        matrix_row = self._correlation_matrix.get(ticker, {})
        if not matrix_row:
            return 1.0

        defense_positions = [
            symbol.upper()
            for symbol in positions.keys()
            if symbol.upper() in self._correlation_matrix and symbol.upper() != ticker
        ]
        if not defense_positions:
            return 1.0

        corr_sum = sum(matrix_row.get(existing, 0.0) for existing in defense_positions)
        # Proportional reduction: each 0.7 correlation contributes a 10.5% haircut.
        reduction = min(0.75, corr_sum * 0.15)
        adjustment = max(0.25, 1.0 - reduction)

        logger.info(
            "correlation_adjustment_applied",
            ticker=ticker,
            correlated_positions=defense_positions,
            corr_sum=round(corr_sum, 3),
            reduction=round(reduction, 3),
            adjustment_multiplier=round(adjustment, 3),
        )

        return adjustment

    def _build_defense_correlation_matrix(self) -> dict[str, dict[str, float]]:
        """Hardcoded pairwise correlation matrix (~0.7) for 12 defense tickers."""
        tickers = [
            "LMT",
            "RTX",
            "NOC",
            "GD",
            "LHX",
            "BA",
            "HII",
            "TDG",
            "LDOS",
            "SAIC",
            "KTOS",
            "PLTR",
        ]
        matrix: dict[str, dict[str, float]] = {}
        for ticker in tickers:
            matrix[ticker] = {}
            for other in tickers:
                matrix[ticker][other] = 1.0 if ticker == other else 0.7
        return matrix
