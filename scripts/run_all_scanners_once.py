"""Run all scanners once to populate fact store and collect statistics."""
from __future__ import annotations

import asyncio
import sys

import structlog
from dotenv import load_dotenv

from src.config.loader import load_config_bundle
from src.logging_setup import configure_logging
from src.processing import ImportanceScorer, NERProcessor, SentimentAnalyzer
from src.scanners.factory import build_scanner
from src.storage import ChromaFactStore, SQLiteRepository

logger = structlog.get_logger(__name__)


async def main() -> None:
    """Run all enabled scanners once and report statistics."""
    bundle = load_config_bundle("config")
    configure_logging(bundle.settings)

    repository = SQLiteRepository("data/trading.db")
    fact_store = ChromaFactStore("data/chroma")
    sentiment = SentimentAnalyzer()
    ner = NERProcessor()
    scorer = ImportanceScorer(bundle.settings, fact_store)

    print("\n" + "=" * 80)
    print("SCANNER BATCH RUN - POPULATING FACT STORE")
    print("=" * 80 + "\n")

    scanner_results = {}
    total_events_queued = 0

    for source in bundle.sources.sources:
        if not source.enabled:
            print(f"⊘ {source.name}: DISABLED (skipped)")
            continue

        print(f"▶ {source.name}: Starting scan...")

        try:
            scanner = build_scanner(
                source=source,
                settings=bundle.settings,
                repository=repository,
                fact_store=fact_store,
                sentiment=sentiment,
                ner=ner,
                scorer=scorer,
            )
            if scanner is None:
                print(f"  ✗ Failed to build scanner for {source.name}")
                continue

            queued = await scanner.run()
            scanner_results[source.name] = len(queued)
            total_events_queued += len(queued)
            print(f"  ✓ {source.name}: {len(queued)} events queued")

        except Exception as exc:
            print(f"  ✗ {source.name} error: {exc}")
            scanner_results[source.name] = 0

    print("\n" + "-" * 80)
    print("SCANNER RESULTS SUMMARY")
    print("-" * 80)

    for source_name, count in sorted(scanner_results.items()):
        status_icon = "✓" if count > 0 else "✗"
        print(f"{status_icon} {source_name}: {count} events")

    print(f"\nTotal events queued: {total_events_queued}")

    # Get database statistics
    print("\n" + "-" * 80)
    print("DATABASE STATISTICS")
    print("-" * 80)

    # Count events in SQLite
    sqlite_count = repository.get_event_count()
    print(f"✓ SQLite events stored: {sqlite_count}")

    # Count embeddings in ChromaDB
    try:
        chroma_count = fact_store._collection.count()
        print(f"✓ ChromaDB embeddings stored: {chroma_count}")
    except Exception as e:
        print(f"✗ ChromaDB count failed: {e}")
        chroma_count = 0

    print("\n" + "=" * 80)
    print(f"POPULATION COMPLETE: {sqlite_count} events, {chroma_count} embeddings")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
