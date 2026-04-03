# Execution Module Architecture

## Overview

The execution module (`src/execution/`) coordinates all order placement, risk management, and position tracking. It bridges the gap between agent proposals and actual broker execution via IBKR.

## Core Design Principles

1. **No Execution Without Risk Check**: Every trade proposal must pass risk validation before order placement.
2. **Bracket Orders for All Trades**: Entry orders are always paired with stop-loss and take-profit exit orders.
3. **Kelly Criterion Position Sizing**: Position size is calculated based on stop-loss distance and account risk allocation.
4. **Hard Limits Are Enforced in Python**: No LLM can override position size, daily loss limits, or concentration checks.
5. **Observable Execution**: Every decision (approved, rejected, executed) is logged with full reasoning.

## Module Structure

```
src/execution/
├── __init__.py        # Public API exports
├── ibkr.py            # IBKR connector (mock + live modes)
├── orders.py          # OrderManager & BracketOrder
├── risk.py            # RiskManager with hard limits
└── executor.py        # TradeExecutor orchestrator
```

## Key Classes

### IBKRConnector
**File**: `ibkr.py`

Abstraction layer over IB Gateway or mock trading.

**Public Methods**:
- `connect()` - Connect to IBKR Gateway or start mock mode
- `place_order()` - Place a limit/market/stop order
- `get_positions()` - Get all open positions
- `get_cash_balance()` - Get available cash
- `get_account_value()` - Get total account equity
- `get_quote()` - Get current price for a ticker
- `cancel_order()` - Cancel a pending order

**Mode**:
- `"mock"` (default): Simulates fills immediately at limit price with mock portfolio
- `"live"`: Real IB Gateway connection via `ib_insync`

### BracketOrder
**File**: `orders.py`

Data class representing a composite order structure.

**Structure**:
```
BracketOrder
├── entry_order_id        (placed immediately as limit order)
├── stop_loss_order_id    (placed after entry fills)
└── take_profit_order_id  (placed after entry fills)
```

**Attributes**:
- `ticker` - Stock symbol
- `entry_quantity` - Number of shares
- `entry_price` - Limit price for entry
- `stop_loss_price` - Price level for stop order
- `take_profit_price` - Price level for profit target
- `status` - One of: pending → entry_filled → exit_filled
- `entry_fill_price` - Actual fill price

### OrderManager
**File**: `orders.py`

Manages the lifecycle of bracket orders.

**Public Methods**:
- `place_bracket_order()` - Create and place an entry order with pre-calculated exits
  - Calculates stop and profit prices based on entry price and configured percentages
  - For long: stop = entry × (1 - stop_loss_pct%), profit = entry × (1 + take_profit_pct%)
  - For short: stop = entry × (1 + stop_loss_pct%), profit = entry × (1 - take_profit_pct%)
  
- `monitor_brackets()` - Check order fills and create exit legs
  - When entry fills → creates stop-loss and take-profit orders
  - When exit fills → cancels the other exit order
  
- `close_position()` - Manually close a position with a limit order

- `validate_proposal_for_execution()` - Pre-checks proposal structure

**Bracket Order Lifecycle**:
```
1. place_bracket_order() called with ticker, side, quantity, entry_price, stop_%,profit_%
   → Entry order placed as limit order
   → BracketOrder created with status="pending"

2. monitor_brackets() detects entry filled
   → Stop-loss order placed as stop order
   → Take-profit order placed as limit order
   → status changed to "entry_filled"

3. monitor_brackets() detects exit filled (stop OR profit)
   → Other exit order cancelled immediately
   → status changed to "exit_filled"
   → Bracket removed from active tracking
```

### RiskManager
**File**: `risk.py`

Enforces hard limits from `config/risk.yaml`.

**Configuration** (from `RiskConfig`):
- `max_position_pct` - Single position cap as % of account
- `max_daily_loss_pct` - Daily realized loss limit
- `max_open_positions` - Concurrent position limit
- `max_sector_concentration_pct` - Sector exposure cap
- `max_trades_per_day` - Trade execution limit
- `cool_down_minutes` - Minutes before same ticker can trade again
- `min_conviction_score` - Min agent conviction (1-10)
- `min_ragas_score` - Min evaluator faithfulness score
- `allowed_tickers` - Whitelist of tradeable symbols

**Public Methods**:
- `calculate_position_size()` - Determine shares using Kelly-inspired logic
  - Input: proposal (with stop_loss_pct), current_price, account_value
  - Formula: shares = (account × risk_pct) / (entry_price × stop_loss_pct%)
  - Capped by max_position_pct limit
  - Returns: `PositionSizing` with calculated shares, risk amount, etc.

- `check_proposal()` - Run all risk checks on proposal
  - Conviction score check
  - RAGAS score check
  - Whitelist check
  - Position count check
  - Trade frequency check
  - Cool-down check
  - Concentration check
  
- `register_trade()` - Log a trade for cool-down tracking

- `validate_trade_for_risk()` - Full validation with position sizing

**Hard Limits Philosophy**:
- These are **not suggestions** — they are enforced in Python code
- LLM cannot negotiate or override them
- Every failed limit is logged with the proposal details

### TradeExecutor
**File**: `executor.py`

Orchestrator that coordinates risk checks → position sizing → order placement.

**Public Methods**:
- `execute_proposal()` - Full trade execution pipeline
  ```
  1. Validate proposal structure (direction, stop_loss_pct, etc.)
  2. Run risk checks (conviction, whitelisted, no open positions left, etc.)
  3. Calculate position size (shares, risk amount, etc.)
  4. Place bracket order via OrderManager
  5. Return ExecutionResult (success/fail + order_id/error_reasons)
  ```
  Returns: `ExecutionResult` for logging and decision tracking

- `monitor_and_update_positions()` - Called periodically (e.g., every 60s)
  - Calls `OrderManager.monitor_brackets()` to check for fills
  - Updates market prices for all open positions
  - Recalculates P&L

- `close_position()` - Manually close a position
  - Required for emergency exits or manual overrides

- `get_portfolio_summary()` - Current portfolio state
  - Account value, cash, invested amount, position count
  - Active bracket statuses

- `get_execution_log()` - Query execution history

**Execution Pipeline** (in `execute_proposal()`):
```
TradeProposal
    ↓
[OrderManager.validate_proposal_for_execution]
    ↓ Pass
[RiskManager.validate_trade_for_risk]
    ↓ Pass
[RiskManager.calculate_position_size] → PositionSizing
    ↓
[OrderManager.place_bracket_order] → order_id
    ↓
[RiskManager.register_trade] (for cool-down tracking)
    ↓
ExecutionResult(success=True, order_id=..., shares=..., entry_price=...)
```

If any step fails, returns `ExecutionResult(success=False, error=..., risk_checks=[reasons])`

## Data Models

### ExecutionResult
```python
ExecutionResult(
    success: bool,
    ticker: str,
    order_id: str | None,     # If successful
    shares: int | None,        # Position size
    entry_price: float | None, # Execution price
    error: str | None,         # If failed
    risk_checks: list[str],    # Rejection reasons
)
```

### PositionSizing
```python
PositionSizing(
    symbol: str,
    direction: str,             # "long" or "short"
    shares: int,
    account_pct: float,         # Position as % of account
    risk_amount_usd: float,     # Max loss if stop hit
    risk_pct_of_capital: float, # Risk as % of capital
    entry_price: float,
    stop_loss_price: float,
)
```

### BracketOrder
```python
BracketOrder(
    entry_order_id: str,
    stop_loss_order_id: str | None,
    take_profit_order_id: str | None,
    ticker: str,
    entry_quantity: int,
    entry_price: float,
    stop_loss_price: float,
    take_profit_price: float,
    status: str,  # pending | entry_filled | exit_filled
    entry_fill_price: float | None,
)
```

## Integration Points

### With Agent Module
1. Agents generate `TradeProposal` objects after discussion
2. Evaluator sets `ragas_score` on proposal
3. Executor receives proposal with all fields populated
4. Execution result logged for auditing

### With Storage Module
- SQLite repo stores `ExecutionResult` history
- `Position` snapshots saved at EOD
- `BracketOrder` lifecycle logged for post-mortem

### With LLM Module
- Risk manager reads `min_conviction_score` from config
- Executor logs context for future agent training/evaluation

## Execution Flow Diagram

```
Agent Discussion
    ↓
TradeProposal {ticker, direction, conviction, stop_loss_pct, take_profit_pct}
    ↓
[Evaluator] 
    ↓
TradeProposal {+ragas_score}
    ↓
[TradeExecutor.execute_proposal()]
    ├─→ Validate structure ✓
    ├─→ Risk Manager checks:
    │   ├─ Conviction score ✓
    │   ├─ RAGAS score ✓
    │   ├─ Ticker whitelisted ✓
    │   ├─ Open position limit ✓
    │   ├─ Trade frequency ✓
    │   └─ Cool-down period ✓
    ├─→ Position sizing: shares = (account × 1%) / (price × stop_loss_pct%)
    ├─→ Place bracket order:
    │   ├─ Entry: limit @ market_price
    │   └─ Pending exit orders ready
    └─→ ExecutionResult {success=True, order_id, shares}
        ↓
    [OrderManager.monitor_brackets()] // Every 60s
        ├─→ Entry filled? Create stop + profit orders
        └─→ Exit filled? Cancel other leg
        
    ↓
Position tracking + P&L calculations
```

## Mock Mode Behavior

When `IBKRConnector(mode="mock")`:
1. `connect()` returns immediately, no network call
2. `place_order()` creates `OrderRecord` and immediately calls `_mock_fill_order()`
3. `_mock_fill_order()` fills order at limit price with mock cash/position updates
4. `get_positions()` returns mock portfolio state
5. `get_account_value()` returns `cash_balance + (position_qty × market_price)`

**Use**: Paper trading, testing, demo, CI/CD

## Logging

All execution decisions use `structlog`:
- `bracket_order_created` - Entry order placed
- `bracket_entry_filled` - Entry filled, exits created
- `bracket_stop_loss_filled` - Stop triggered (loss taken)
- `bracket_take_profit_filled` - Profit target hit (win taken)
- `position_closed` - Manual close completed
- `proposal_validation_failed` - Structure error (rare)
- `risk_validation_failed` - Hard limit violated
- `order_placement_error` - Broker error (connectivity, symbol not found, etc.)

## Testing Strategy

**Unit Tests** (`tests/test_execution.py`):
- Order record creation and filling
- Position P&L calculation
- Bracket order creation and status transitions
- Risk manager position sizing
- Executor pipeline behavior
- Portfolio summary queries

**Integration Tests** (manual):
- Mock mode: start broker, place trade, monitor, close
- Live mode: connect to IB Gateway, place small order, verify

**Load Tests**:
- 1000 order/minute throughput
- 50 concurrent bracket orders
- Market data updates at 100Hz

## Future Enhancements

1. **OCO Orders** - Use native IBKR OCO (one-cancels-other) if available
2. **Partial Fills** - Handle partial entry fills by scaling exit sizes
3. **Dynamic Stop Adjustment** - Trailing stops based on volatility
4. **ML-based Position Sizing** - Predict win rates, adjust Kelly fraction
5. **Hedge Orders** - Automatically hedge large sector concentrations
6. **Port to Cloud** - Run executor on AWS Lambda for ultra-low latency
