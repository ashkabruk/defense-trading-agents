"""Connect to IB Gateway paper port 4002, print account summary and LMT quote, then disconnect."""
from __future__ import annotations

import asyncio

from src.execution.ibkr import IBKRConnector


async def main() -> None:
    connector = IBKRConnector(mode="paper", host="127.0.0.1", port=4002, client_id=101)

    print("Connecting to IB Gateway at 127.0.0.1:4002 ...")
    connected = await connector.connect()
    if not connected:
        print("Failed to connect.")
        return

    try:
        print("Connected.")
        print("\n=== Account Summary ===")
        summary_rows = await connector.get_account_summary()
        for row in summary_rows:
            print(f"{row['account']} | {row['tag']} | {row['value']} {row['currency']}")

        buying_power = await connector.get_buying_power()
        print("\n=== Buying Power ===")
        print(f"{buying_power:.2f}")

        print("\n=== LMT Quote (last/close fallback) ===")
        try:
            quote = await connector.get_quote("LMT")
            print(f"LMT last price used for order pricing: {quote}")
        except Exception as exc:
            print(f"LMT quote unavailable: {exc}")
    finally:
        await connector.disconnect()
        print("\nDisconnected.")


if __name__ == "__main__":
    asyncio.run(main())
