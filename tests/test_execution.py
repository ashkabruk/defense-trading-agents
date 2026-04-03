"""Tests for execution module."""
import pytest
from datetime import datetime

from src.execution.ibkr import IBKRConnector, OrderRecord, OrderStatus, OrderType, Position
from src.execution.orders import OrderManager, BracketOrder
from src.execution.risk import RiskManager, RiskCheckResult
from src.execution.executor import TradeExecutor, ExecutionResult
from src.models.core import TradeProposal
from src.models.config import RiskConfig


@pytest.fixture
def ibkr_connector():
    """Mock IBKR connector fixture."""
    return IBKRConnector(mode="mock")


@pytest.fixture
async def connected_ibkr(ibkr_connector):
    """Connected IBKR connector fixture."""
    await ibkr_connector.connect()
    yield ibkr_connector
    await ibkr_connector.disconnect()


@pytest.fixture
def order_manager(connected_ibkr):
    """Order manager fixture."""
    return OrderManager(connected_ibkr)


@pytest.fixture
def risk_config():
    """Risk config fixture."""
    return RiskConfig(
        max_position_pct=5.0,
        max_daily_loss_pct=2.0,
        max_open_positions=5,
        max_trades_per_day=10,
        cool_down_minutes=30,
        min_conviction_score=0.6,
        min_ragas_score=0.5,
        max_sector_concentration_pct=15.0,
        max_holding_days=30,
        allowed_tickers=["LMT", "RTX", "NOC", "GDIS"],
    )


@pytest.fixture
def risk_manager(risk_config, connected_ibkr):
    """Risk manager fixture."""
    return RiskManager(risk_config, connected_ibkr)


@pytest.fixture
def executor(connected_ibkr, order_manager, risk_manager):
    """Trade executor fixture."""
    return TradeExecutor(connected_ibkr, order_manager, risk_manager)


class TestOrderRecord:
    """Test OrderRecord data class."""

    def test_order_record_creation(self):
        """Test creating an order record."""
        order = OrderRecord(
            order_id="ORD-000001",
            ticker="LMT",
            order_type=OrderType.LIMIT,
            side="BUY",
            quantity=100,
            limit_price=350.0,
            status=OrderStatus.SUBMITTED,
        )

        assert order.order_id == "ORD-000001"
        assert order.ticker == "LMT"
        assert order.order_type == OrderType.LIMIT
        assert order.side == "BUY"
        assert order.quantity == 100
        assert order.limit_price == 350.0
        assert order.filled_qty == 0

    def test_order_record_fill(self):
        """Test filling an order."""
        order = OrderRecord(
            order_id="ORD-000001",
            ticker="LMT",
            order_type=OrderType.LIMIT,
            side="BUY",
            quantity=100,
            limit_price=350.0,
        )

        order.filled_qty = 100
        order.filled_price = 349.50
        order.status = OrderStatus.FILLED

        assert order.status == OrderStatus.FILLED
        assert order.filled_qty == 100
        assert order.filled_price == 349.50


class TestPosition:
    """Test Position data class."""

    def test_position_creation(self):
        """Test creating a position."""
        pos = Position(
            ticker="LMT",
            quantity=100,
            entry_price=350.0,
            current_price=351.0,
            stop_loss_price=340.0,
        )

        assert pos.ticker == "LMT"
        assert pos.quantity == 100
        assert pos.entry_price == 350.0
        assert pos.stop_loss_price == 340.0

    def test_position_pnl_calculation(self):
        """Test P&L calculation."""
        pos = Position(
            ticker="LMT",
            quantity=100,
            entry_price=350.0,
        )

        pos.update_market_price(355.0)

        assert pos.current_price == 355.0
        assert pos.pnl_pct == pytest.approx(1.43, abs=0.01)


@pytest.mark.asyncio
async def test_bracket_order_creation(order_manager):
    """Test creating a bracket order."""
    order_id = await order_manager.place_bracket_order(
        ticker="LMT",
        side="BUY",
        quantity=100,
        entry_limit_price=350.0,
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
    )

    assert order_id is not None
    assert order_id in order_manager.brackets
    bracket = order_manager.brackets[order_id]
    assert bracket.ticker == "LMT"
    assert bracket.entry_quantity == 100
    assert bracket.entry_price == 350.0
    assert bracket.stop_loss_price == pytest.approx(343.0, abs=0.01)
    assert bracket.take_profit_price == pytest.approx(364.0, abs=0.01)


@pytest.mark.asyncio
async def test_bracket_order_status_transitions(order_manager, connected_ibkr):
    """Test bracket order status transitions."""
    order_id = await order_manager.place_bracket_order(
        ticker="RTX",
        side="BUY",
        quantity=50,
        entry_limit_price=100.0,
        stop_loss_pct=2.0,
        take_profit_pct=3.0,
    )

    bracket = order_manager.brackets[order_id]
    assert bracket.status == "pending"

    # Simulate entry fill
    bracket.status = "entry_filled"
    bracket.entry_fill_price = 99.99

    assert bracket.status == "entry_filled"
    assert bracket.entry_fill_price == 99.99


@pytest.mark.asyncio
async def test_ibkr_connector_mock_mode(connected_ibkr):
    """Test IBKR connector in mock mode."""
    assert connected_ibkr.is_connected
    assert connected_ibkr.mode == "mock"

    account_value = await connected_ibkr.get_account_value()
    assert account_value == 100000.0


@pytest.mark.asyncio
async def test_risk_manager_position_sizing(risk_manager, connected_ibkr):
    """Test position sizing calculation."""
    proposal = TradeProposal(
        ticker="LMT",
        direction="long",
        conviction=0.85,
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        position_size_pct=3.0,
        source="test",
    )

    account_value = await connected_ibkr.get_account_value()
    sizing = await risk_manager.calculate_position_size(proposal, 350.0, account_value)

    assert sizing is not None
    assert sizing.symbol == "LMT"
    assert sizing.shares > 0
    assert sizing.account_pct <= risk_manager.limits.max_position_pct


@pytest.mark.asyncio
async def test_executor_execution_result(executor):
    """Test ExecutionResult creation."""
    result = ExecutionResult(
        success=True,
        ticker="LMT",
        order_id="ORD-000001",
        shares=100,
        entry_price=350.0,
    )

    assert result.success
    assert result.ticker == "LMT"
    assert result.order_id == "ORD-000001"
    assert result.error is None


@pytest.mark.asyncio
async def test_executor_portfolio_summary(executor):
    """Test getting portfolio summary."""
    summary = await executor.get_portfolio_summary()

    assert "account_value" in summary
    assert "cash" in summary
    assert "invested" in summary
    assert "position_count" in summary
    assert "active_brackets" in summary
    assert summary["account_value"] == 100000.0
