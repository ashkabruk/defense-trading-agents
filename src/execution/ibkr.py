"""IBKR connector with mock mode, paper/live support, and auto-reconnect."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from math import isnan
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class OrderStatus(str, Enum):
    """Order status enum."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(str, Enum):
    """Order type enum."""

    LIMIT = "limit"
    MARKET = "market"
    STOP = "stop"
    TRAILING_STOP = "trailing_stop"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


@dataclass(slots=True)
class OrderRecord:
    """Single order record for tracking."""

    order_id: str
    ticker: str
    order_type: OrderType
    side: str
    quantity: int
    limit_price: float | None = None
    stop_price: float | None = None
    trailing_amount: float | None = None
    trailing_percent: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    filled_price: float | None = None
    timestamp_submitted: str | None = None
    timestamp_filled: str | None = None


@dataclass(slots=True)
class Position:
    """Active position tracking."""

    ticker: str
    quantity: int
    entry_price: float
    current_price: float = 0.0
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    entry_time: str | None = None
    close_time: str | None = None
    pnl_pct: float = 0.0

    def update_market_price(self, new_price: float) -> None:
        """Update current market price and recalculate P&L."""
        self.current_price = new_price
        if self.entry_price <= 0:
            return
        if self.quantity >= 0:
            self.pnl_pct = ((new_price - self.entry_price) / self.entry_price) * 100
        else:
            self.pnl_pct = ((self.entry_price - new_price) / self.entry_price) * 100


class IBKRConnector:
    """IBKR connection with mock/paper/live support and reconnect handling."""

    def __init__(
        self,
        mode: str = "paper",
        host: str = "127.0.0.1",
        port: int | None = None,
        client_id: int = 1,
        reconnect_attempts: int = 3,
        reconnect_delay_seconds: float = 2.0,
    ) -> None:
        self.mode = mode
        self.host = host
        self.client_id = client_id
        self.port = self._resolve_port(mode, port)

        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_seconds = reconnect_delay_seconds

        self.is_connected = False
        self.ib_client: Any | None = None

        self.positions: dict[str, Position] = {}
        self.orders: dict[str, OrderRecord] = {}
        self._live_trades: dict[str, Any] = {}
        self.cash_balance: float = 100000.0
        self.market_data: dict[str, float] = {}
        self.order_counter: int = 1

        logger.info(
            "connector_initialized",
            mode=self.mode,
            host=self.host,
            port=self.port,
            client_id=self.client_id,
        )

    @staticmethod
    def _resolve_port(mode: str, port: int | None) -> int:
        if port is not None:
            return port
        if mode == "paper":
            return 4002
        if mode == "live":
            return 4001
        return 7497

    async def connect(self) -> bool:
        """Connect to mock or IB Gateway."""
        if self.mode == "mock":
            self.is_connected = True
            logger.info("connector_mock_connected")
            return True

        try:
            from ib_insync import IB

            if self.ib_client is None:
                self.ib_client = IB()
                self.ib_client.disconnectedEvent += self._on_disconnect
            else:
                # Ensure previous session is fully released before reconnecting
                await asyncio.sleep(1.0)

            await self.ib_client.connectAsync(
                self.host,
                self.port,
                clientId=self.client_id,
                timeout=10,
            )
            self.ib_client.reqMarketDataType(3)
            self.is_connected = bool(self.ib_client.isConnected())

            logger.info(
                "connector_live_connected",
                host=self.host,
                port=self.port,
                mode=self.mode,
                connected=self.is_connected,
            )
            return self.is_connected
        except asyncio.CancelledError as exc:
            self.is_connected = False
            logger.error("connector_connection_cancelled", error=str(exc), mode=self.mode)
            return False
        except Exception as exc:
            self.is_connected = False
            logger.error("connector_connection_failed", error=str(exc), mode=self.mode)
            return False

    def _on_disconnect(self) -> None:
        self.is_connected = False
        logger.warning("connector_disconnected_event", mode=self.mode)

    async def ensure_connected(self) -> None:
        """Ensure connection is alive; reconnect automatically if dropped."""
        if self.mode == "mock":
            self.is_connected = True
            return

        client_ok = self.ib_client is not None and self.ib_client.isConnected()
        if self.is_connected and client_ok:
            return

        for attempt in range(1, self.reconnect_attempts + 1):
            connected = await self.connect()
            if connected:
                logger.info("connector_reconnected", attempt=attempt)
                return
            await asyncio.sleep(self.reconnect_delay_seconds)

        raise RuntimeError("Unable to reconnect to IB Gateway")

    async def disconnect(self) -> None:
        """Disconnect from IBKR Gateway."""
        if self.mode != "mock" and self.ib_client is not None:
            try:
                self.ib_client.disconnect()
            except Exception as exc:
                logger.error("connector_disconnect_error", error=str(exc))
        self.is_connected = False
        logger.info("connector_disconnected", mode=self.mode)

    async def get_quote(self, ticker: str) -> float:
        """Return latest usable last price (fallback close) for a ticker."""
        if self.mode == "mock":
            return self.market_data.get(ticker, 100.0)

        from ib_insync import Stock

        last_error: Exception | None = None
        for _ in range(self.reconnect_attempts):
            await self.ensure_connected()

            try:
                contract = Stock(ticker, "SMART", "USD")
                await self.ib_client.qualifyContractsAsync(contract)

                tick = self.ib_client.reqMktData(contract, "", False, False)
                await asyncio.sleep(2.0)

                price = self._coerce_price(getattr(tick, "last", None))
                if price is None:
                    price = self._coerce_price(getattr(tick, "close", None))

                self.ib_client.cancelMktData(contract)

                if price is None:
                    bars = await self.ib_client.reqHistoricalDataAsync(
                        contract,
                        endDateTime="",
                        durationStr="1 D",
                        barSizeSetting="1 day",
                        whatToShow="TRADES",
                        useRTH=True,
                    )
                    if bars:
                        price = self._coerce_price(getattr(bars[-1], "close", None))

                if price is not None and price > 0:
                    self.market_data[ticker] = price
                    return price
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(self.reconnect_delay_seconds)

        cached = self.market_data.get(ticker)
        if cached is not None and cached > 0:
            return cached

        if last_error is not None:
            raise RuntimeError(f"No usable quote for {ticker}: {last_error}") from last_error
        raise RuntimeError(f"No usable quote for {ticker}")

    @staticmethod
    def _coerce_price(value: Any) -> float | None:
        try:
            if value is None:
                return None
            as_float = float(value)
            if isnan(as_float):
                return None
            return as_float
        except (TypeError, ValueError):
            return None

    async def place_order(
        self,
        ticker: str,
        side: str,
        quantity: int,
        limit_price: float | None = None,
        stop_price: float | None = None,
        trailing_amount: float | None = None,
        trailing_percent: float | None = None,
        market_order: bool = False,
    ) -> str:
        """Place an order.

        If no limit price is supplied for entry/exit limit orders, uses latest last price.
        """
        if quantity <= 0:
            raise ValueError("quantity must be > 0")

        side = side.upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")

        if trailing_amount is not None or trailing_percent is not None:
            order_type = OrderType.TRAILING_STOP
        elif stop_price is not None:
            order_type = OrderType.STOP
        elif market_order:
            order_type = OrderType.MARKET
        elif limit_price is not None:
            order_type = OrderType.LIMIT
        else:
            order_type = OrderType.MARKET

        if self.mode == "mock":
            order_id = f"ORD-{self.order_counter:06d}"
            self.order_counter += 1
            if order_type == OrderType.LIMIT and limit_price is None:
                limit_price = self.market_data.get(ticker, 100.0)

            record = OrderRecord(
                order_id=order_id,
                ticker=ticker,
                order_type=order_type,
                side=side,
                quantity=quantity,
                limit_price=limit_price,
                stop_price=stop_price,
                trailing_amount=trailing_amount,
                trailing_percent=trailing_percent,
                status=OrderStatus.SUBMITTED,
            )
            self.orders[order_id] = record

            # Entry/close orders fill immediately in mock mode; stop/trailing legs stay pending.
            if order_type in {OrderType.LIMIT, OrderType.MARKET}:
                await self._mock_fill_order(order_id)
            return order_id

        await self.ensure_connected()
        from ib_insync import LimitOrder, MarketOrder, Order, Stock, StopOrder

        if order_type == OrderType.LIMIT and limit_price is None:
            limit_price = await self.get_quote(ticker)

        contract = Stock(ticker, "SMART", "USD")
        await self.ib_client.qualifyContractsAsync(contract)

        if order_type == OrderType.TRAILING_STOP:
            ib_order = Order(orderType="TRAIL", action=side, totalQuantity=quantity)
            if trailing_amount is not None:
                ib_order.auxPrice = float(trailing_amount)
            if trailing_percent is not None:
                ib_order.trailingPercent = float(trailing_percent)
            if stop_price is not None:
                ib_order.trailStopPrice = float(stop_price)
        elif order_type == OrderType.STOP:
            ib_order = StopOrder(side, quantity, stop_price or 0.0)
        elif order_type == OrderType.MARKET:
            ib_order = MarketOrder(side, quantity)
        else:
            ib_order = LimitOrder(side, quantity, limit_price or 0.0)

        trade = self.ib_client.placeOrder(contract, ib_order)
        await asyncio.sleep(0.2)

        broker_id = getattr(trade.order, "orderId", None)
        if broker_id is None:
            order_id = f"IB-{self.order_counter:06d}"
            self.order_counter += 1
        else:
            order_id = str(broker_id)

        record = OrderRecord(
            order_id=order_id,
            ticker=ticker,
            order_type=order_type,
            side=side,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            trailing_amount=trailing_amount,
            trailing_percent=trailing_percent,
            status=self._map_status(getattr(trade.orderStatus, "status", "Submitted")),
            filled_qty=int(getattr(trade.orderStatus, "filled", 0) or 0),
            filled_price=self._coerce_price(getattr(trade.orderStatus, "avgFillPrice", None)),
        )
        self.orders[order_id] = record
        self._live_trades[order_id] = trade

        logger.info(
            "order_placed",
            order_id=order_id,
            ticker=ticker,
            side=side,
            quantity=quantity,
            order_type=order_type.value,
            mode=self.mode,
        )
        return order_id

    async def _mock_fill_order(self, order_id: str) -> None:
        from datetime import datetime

        order = self.orders.get(order_id)
        if order is None:
            return

        market_price = self.market_data.get(order.ticker, 100.0)
        fill_price = order.limit_price if order.limit_price is not None else market_price

        order.filled_price = fill_price
        order.filled_qty = order.quantity
        order.status = OrderStatus.FILLED
        order.timestamp_filled = datetime.utcnow().isoformat()

        self._apply_mock_fill_to_positions(order)

    def _apply_mock_fill_to_positions(self, order: OrderRecord) -> None:
        signed_qty = order.quantity if order.side == "BUY" else -order.quantity
        fill_price = order.filled_price or self.market_data.get(order.ticker, 100.0)

        current = self.positions.get(order.ticker)
        current_qty = current.quantity if current else 0
        new_qty = current_qty + signed_qty

        if new_qty == 0:
            if order.ticker in self.positions:
                del self.positions[order.ticker]
        else:
            if current is None:
                self.positions[order.ticker] = Position(
                    ticker=order.ticker,
                    quantity=new_qty,
                    entry_price=fill_price,
                    current_price=fill_price,
                )
            else:
                if (current_qty > 0 and signed_qty > 0) or (current_qty < 0 and signed_qty < 0):
                    total_abs_qty = abs(current_qty) + abs(signed_qty)
                    weighted = (current.entry_price * abs(current_qty)) + (fill_price * abs(signed_qty))
                    current.entry_price = weighted / total_abs_qty
                current.quantity = new_qty
                current.current_price = fill_price
                current.update_market_price(fill_price)

        cash_change = fill_price * order.quantity
        if order.side == "BUY":
            self.cash_balance -= cash_change
        else:
            self.cash_balance += cash_change

    async def get_positions(self) -> dict[str, Position]:
        """Get current positions."""
        if self.mode == "mock":
            return self.positions.copy()

        await self.ensure_connected()
        out: dict[str, Position] = {}
        for pos in self.ib_client.positions():
            contract = pos.contract
            qty = int(pos.position)
            if qty == 0:
                continue

            current_price = self.market_data.get(contract.symbol, 0.0)
            p = Position(
                ticker=contract.symbol,
                quantity=qty,
                entry_price=float(pos.avgCost or 0.0),
                current_price=current_price,
            )
            if current_price > 0:
                p.update_market_price(current_price)
            out[contract.symbol] = p
        return out

    async def get_orders(self) -> dict[str, OrderRecord]:
        """Get all known orders, refreshing live order states."""
        if self.mode != "mock":
            for order_id, trade in self._live_trades.items():
                record = self.orders.get(order_id)
                if record is None:
                    continue
                record.status = self._map_status(getattr(trade.orderStatus, "status", "Submitted"))
                record.filled_qty = int(getattr(trade.orderStatus, "filled", 0) or 0)
                avg_fill = self._coerce_price(getattr(trade.orderStatus, "avgFillPrice", None))
                if avg_fill is not None and avg_fill > 0:
                    record.filled_price = avg_fill
        return self.orders.copy()

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        order = self.orders.get(order_id)
        if order is None:
            return False

        if order.status in {OrderStatus.FILLED, OrderStatus.CANCELLED}:
            return False

        if self.mode != "mock":
            await self.ensure_connected()
            trade = self._live_trades.get(order_id)
            if trade is not None:
                self.ib_client.cancelOrder(trade.order)

        order.status = OrderStatus.CANCELLED
        logger.info("order_cancelled", order_id=order_id)
        return True

    async def get_account_summary(self) -> list[dict[str, str]]:
        """Return account summary rows."""
        if self.mode == "mock":
            return [
                {
                    "account": "MOCK",
                    "tag": "NetLiquidation",
                    "value": f"{self.cash_balance:.2f}",
                    "currency": "USD",
                },
                {
                    "account": "MOCK",
                    "tag": "BuyingPower",
                    "value": f"{self.cash_balance * 4:.2f}",
                    "currency": "USD",
                },
            ]

        await self.ensure_connected()
        out: list[dict[str, str]] = []
        rows = await self.ib_client.accountSummaryAsync()
        for row in rows:
            out.append(
                {
                    "account": row.account,
                    "tag": row.tag,
                    "value": row.value,
                    "currency": row.currency,
                }
            )
        return out

    async def get_buying_power(self) -> float:
        """Return current buying power in account base currency."""
        if self.mode == "mock":
            return self.cash_balance * 4

        rows = await self.get_account_summary()
        for row in rows:
            if row["tag"] == "BuyingPower":
                value = self._coerce_price(row["value"])
                if value is not None:
                    return value
        return 0.0

    async def get_cash_balance(self) -> float:
        """Get available cash in account base currency."""
        if self.mode == "mock":
            return self.cash_balance

        rows = await self.get_account_summary()
        for row in rows:
            if row["tag"] in {"TotalCashValue", "CashBalance"} and row["currency"] in {"USD", "BASE"}:
                value = self._coerce_price(row["value"])
                if value is not None:
                    return value
        return 0.0

    async def get_account_value(self) -> float:
        """Get net liquidation / account equity value."""
        if self.mode == "mock":
            invested = 0.0
            for pos in self.positions.values():
                price = pos.current_price or pos.entry_price
                invested += abs(pos.quantity) * price
            return self.cash_balance + invested

        rows = await self.get_account_summary()
        for row in rows:
            if row["tag"] == "NetLiquidation" and row["currency"] in {"USD", "BASE"}:
                value = self._coerce_price(row["value"])
                if value is not None:
                    return value
        return 0.0

    @staticmethod
    def _map_status(status_text: str) -> OrderStatus:
        value = status_text.lower()
        if value in {"filled"}:
            return OrderStatus.FILLED
        if value in {"cancelled", "inactive"}:
            return OrderStatus.CANCELLED
        if value in {"pendingcancel", "presubmitted", "submitted", "api pending"}:
            return OrderStatus.SUBMITTED
        if value in {"rejected"}:
            return OrderStatus.REJECTED
        return OrderStatus.PENDING

    async def __aenter__(self) -> IBKRConnector:
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()
