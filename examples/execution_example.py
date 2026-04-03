"""Example usage of the execution module.

This demonstrates a complete execution workflow:
1. Create a trade proposal from agent discussion
2. Run risk checks
3. Execute the trade with bracket orders
4. Monitor and close
"""

import asyncio
from datetime import datetime, timezone

from src.execution import IBKRConnector, OrderManager, RiskManager, TradeExecutor
from src.models import TradeProposal, RiskConfig


async def main():
    """Main execution example."""

    # Step 1: Initialize components
    print("=== Execution Module Example ===\n")

    # Create IBKR connector (mock mode for demo)
    ibkr = IBKRConnector(mode="mock")
    await ibkr.connect()
    print(f"✓ Connected to IBKR in {ibkr.mode} mode")

    # Create risk manager with hard limits
    risk_config = RiskConfig(
        max_position_pct=5.0,
        max_daily_loss_pct=2.0,
        max_open_positions=5,
        max_trades_per_day=10,
        cool_down_minutes=30,
        min_conviction_score=6,
        min_ragas_score=0.50,
        max_sector_concentration_pct=15.0,
        max_holding_days=30,
        allowed_tickers=["LMT", "RTX", "NOC", "GD"],
    )
    risk_manager = RiskManager(risk_config, ibkr)
    print(f"✓ Risk manager initialized with limits from config")

    # Create order manager
    order_manager = OrderManager(ibkr)
    print(f"✓ Order manager initialized")

    # Create executor
    executor = TradeExecutor(ibkr, order_manager, risk_manager)
    print(f"✓ Trade executor initialized\n")

    # Step 2: Create a trade proposal (simulating agent output)
    print("--- Creating Trade Proposal ---")
    proposal = TradeProposal(
        timestamp=datetime.now(timezone.utc),
        ticker="LMT",
        direction="long",
        conviction=8.5,
        entry_rationale="Defense contractor gaining market share in ICBM modernization",
        risk_factors=["Budget uncertainty", "Competition from RTX"],
        position_size_pct=3.5,
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        max_holding_days=14,
        source_events=["defense_contract_1", "earnings_beat_2"],
        ragas_score=0.78,  # From evaluator
    )
    print(f"Proposal: {proposal.ticker} {proposal.direction.upper()}")
    print(f"  Conviction: {proposal.conviction}/10")
    print(f"  Stop loss: {proposal.stop_loss_pct}%")
    print(f"  Take profit: {proposal.take_profit_pct}%")
    print(f"  RAGAS score: {proposal.ragas_score:.2f}\n")

    # Step 3: Execute the proposal
    print("--- Executing Trade ---")
    current_prices = {"LMT": 350.00, "RTX": 110.50, "NOC": 250.00, "GD": 200.00}

    result = await executor.execute_proposal(proposal, current_prices)

    if result.success:
        print(f"✓ Trade EXECUTED")
        print(f"  Order ID: {result.order_id}")
        print(f"  Shares: {result.shares}")
        print(f"  Entry price: ${result.entry_price:.2f}")
    else:
        print(f"✗ Trade REJECTED")
        print(f"  Error: {result.error}")
        if result.risk_checks:
            for reason in result.risk_checks:
                print(f"    - {reason}")
    print()

    # Step 4: Get portfolio summary
    print("--- Portfolio Summary ---")
    summary = await executor.get_portfolio_summary()
    print(f"Account value: ${summary['account_value']:,.2f}")
    print(f"Cash: ${summary['cash']:,.2f}")
    print(f"Invested: ${summary['invested']:,.2f}")
    print(f"Open positions: {summary['position_count']}")
    print(f"Active bracket orders: {summary['active_brackets']}")
    print(f"Bracket statuses: {summary['bracket_statuses']}\n")

    # Step 5: Simulate market movement and monitoring
    print("--- Monitoring Positions (Simulated) ---")
    await executor.monitor_and_update_positions()
    print("✓ Positions updated and monitored\n")

    # Step 6: Execute another proposal to test limits
    print("--- Testing Risk Limits ---")
    proposal2 = TradeProposal(
        timestamp=datetime.now(timezone.utc),
        ticker="RTX",
        direction="short",
        conviction=7.5,
        entry_rationale="Technical breakdown after earnings",
        risk_factors=["Short squeeze risk"],
        position_size_pct=5.5,  # Exceeds max_position_pct of 5.0
        stop_loss_pct=2.0,
        take_profit_pct=3.0,
        max_holding_days=5,
        source_events=[],
        ragas_score=0.65,
    )

    result2 = await executor.execute_proposal(proposal2, current_prices)
    if result2.success:
        print(f"✓ Second trade executed: {result2.order_id}")
    else:
        print(f"✗ Second trade rejected (as expected)")
        if result2.risk_checks:
            for reason in result2.risk_checks:
                print(f"  - {reason}")
    print()

    # Step 7: Access execution log
    print("--- Execution Log ---")
    log = executor.get_execution_log()
    print(f"Total executions: {len(log)}")
    for i, exec_result in enumerate(log, 1):
        status = "✓ SUCCESS" if exec_result.success else "✗ FAILED"
        print(f"  {i}. {exec_result.ticker} - {status}")
        if exec_result.order_id:
            print(f"     Order: {exec_result.order_id}")
    print()

    # Cleanup
    await ibkr.disconnect()
    print("✓ Disconnected from IBKR")


if __name__ == "__main__":
    asyncio.run(main())
