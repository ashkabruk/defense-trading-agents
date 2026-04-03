from __future__ import annotations

import argparse
import asyncio
import signal
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import structlog
from dotenv import load_dotenv

from src.agents.orchestrator import DiscussionOrchestrator
from src.config.loader import ConfigBundle, load_config_bundle
from src.evaluation.ragas import RAGASEvaluator
from src.execution import IBKRConnector, OrderManager, RiskManager, TradeExecutor
from src.llm.client import LLMClient
from src.logging_setup import configure_logging
from src.processing import ImportanceScorer, NERProcessor, SentimentAnalyzer
from src.scanners.factory import build_scanner
from src.storage import ChromaFactStore, SQLiteRepository

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class RuntimeStats:
    """Mutable runtime counters for periodic status reporting."""

    events_scanned: int = 0
    discussions_triggered: int = 0
    trades_placed: int = 0
    proposals_rejected: int = 0
    last_status_at: datetime | None = None
    discussion_times: list[datetime] = field(default_factory=list)


class AutonomousTradingApp:
    """Main autonomous trading loop orchestrating scan -> discuss -> evaluate -> execute."""

    def __init__(self, bundle: ConfigBundle) -> None:
        self.bundle = bundle
        self.settings = bundle.settings
        self.risk_config = bundle.risk

        self.repository = SQLiteRepository("data/trading.db")
        self.fact_store = ChromaFactStore("data/chroma")
        self.sentiment = SentimentAnalyzer()
        self.ner = NERProcessor()
        self.scorer = ImportanceScorer(self.settings, self.fact_store)

        self.llm_client = LLMClient()
        self.orchestrator = DiscussionOrchestrator.from_config(
            config_dir="config",
            repository=self.repository,
            fact_store=self.fact_store,
            llm_client=self.llm_client,
        )
        self.evaluator = RAGASEvaluator(
            llm_client=self.llm_client,
            model_config=self.settings.models["evaluator"],
            fact_store=self.fact_store,
        )

        # Requirement: paper trading with IB Gateway only, client ID 1.
        self.ibkr = IBKRConnector(
            mode="paper",
            host=self.risk_config.ibkr.host,
            port=4002,
            client_id=1,
        )
        self.risk_manager = RiskManager(self.risk_config, self.ibkr)
        self.order_manager = OrderManager(self.ibkr)
        self.executor = TradeExecutor(self.ibkr, self.order_manager, self.risk_manager)

        self.stats = RuntimeStats()
        self.shutdown_event = asyncio.Event()
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self.tasks: list[asyncio.Task] = []
        self.first_run_initialized = False

    async def run(self) -> None:
        """Start all loops and block until shutdown is requested."""
        self._install_signal_handlers()
        await self._connect_broker()

        if not self.first_run_initialized:
            await self._initialize_first_run()

        self._start_background_tasks()

        try:
            await self.shutdown_event.wait()
        finally:
            await self._shutdown()

    async def run_once(self, force_discuss: bool = False) -> None:
        """Run one full cycle: all scanners once, then process queued events once.

        When force_discuss=True, skip first-run warmup and replay top-N recent events
        (by importance) through discuss/evaluate/execute even if they are not new.
        """
        await self._connect_broker()
        try:
            # Production behavior: warm scanners first to avoid backfilling historical items.
            if not force_discuss and not self.first_run_initialized:
                await self._initialize_first_run()
            elif force_discuss:
                logger.info("force_discuss_enabled", warmup_skipped=True)
            
            await self._run_all_scanners_once()
            if force_discuss:
                await self._enqueue_top_recent_events_for_forced_discussion(self.settings.max_queue_size)
            await self._drain_event_queue_once()
            with suppress(Exception):
                await self.executor.monitor_and_update_positions()

            now = datetime.now(timezone.utc)
            print(
                "[STATUS] "
                f"{now.isoformat()} "
                f"events_scanned={self.stats.events_scanned} "
                f"discussions_triggered={self.stats.discussions_triggered} "
                f"trades_placed={self.stats.trades_placed} "
                f"queue_size={self.event_queue.qsize()}"
            )
        finally:
            await self._shutdown()

    def _install_signal_handlers(self) -> None:
        """Register Ctrl+C/termination handlers for graceful shutdown."""

        def _request_shutdown() -> None:
            if not self.shutdown_event.is_set():
                print("\nShutdown signal received. Stopping gracefully...")
                self.shutdown_event.set()

        with suppress(NotImplementedError):
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGINT, _request_shutdown)
            loop.add_signal_handler(signal.SIGTERM, _request_shutdown)

    async def _connect_broker(self) -> None:
        connected = await self.ibkr.connect()
        if connected:
            logger.info("broker_connected", host=self.ibkr.host, port=self.ibkr.port, client_id=1)
        else:
            logger.warning("broker_connect_failed_startup", host=self.ibkr.host, port=self.ibkr.port)

    def _start_background_tasks(self) -> None:
        # Scanner loops (each source on its own configured interval).
        for source in self.bundle.sources.sources:
            if not source.enabled:
                continue

            scanner = build_scanner(
                source=source,
                settings=self.settings,
                repository=self.repository,
                fact_store=self.fact_store,
                sentiment=self.sentiment,
                ner=self.ner,
                scorer=self.scorer,
            )
            if scanner is None:
                continue

            task = asyncio.create_task(self._scanner_loop(source.name, source.interval_seconds, scanner))
            self.tasks.append(task)

        # Discussion/evaluation/execution consumer.
        self.tasks.append(asyncio.create_task(self._event_consumer_loop()))
        # Monitor open brackets/positions.
        self.tasks.append(asyncio.create_task(self._position_monitor_loop()))
        # Print status every 5 minutes.
        self.tasks.append(asyncio.create_task(self._status_loop()))

    async def _scanner_loop(self, source_name: str, interval_seconds: int, scanner) -> None:
        """Repeatedly run one scanner and queue high-importance events."""
        while not self.shutdown_event.is_set():
            try:
                queued_events = await scanner.run()
                self.stats.events_scanned += len(queued_events)

                for event in queued_events:
                    if self.event_queue.full():
                        logger.warning("event_queue_full_drop", source=source_name, event_id=event.id)
                        continue
                    await self.event_queue.put(event)

                logger.info(
                    "scanner_cycle_complete",
                    source=source_name,
                    queued=len(queued_events),
                    queue_size=self.event_queue.qsize(),
                )
            except Exception as exc:
                logger.exception("scanner_loop_error", source=source_name, error=str(exc))

            try:
                await asyncio.wait_for(self.shutdown_event.wait(), timeout=interval_seconds)
            except asyncio.TimeoutError:
                pass

    async def _event_consumer_loop(self) -> None:
        """Consume queued events and run discuss -> evaluate -> execute pipeline."""
        while not self.shutdown_event.is_set():
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            try:
                await self._process_event(event)
            except Exception as exc:
                logger.exception("event_consumer_error", error=str(exc))
            finally:
                self.event_queue.task_done()

    async def _process_event(self, event) -> None:
        """Run one event through discuss -> evaluate -> execute."""
        if not self._can_trigger_discussion_now():
            logger.info("discussion_daily_limit_reached", event_id=event.id)
            return

        self.stats.discussions_triggered += 1
        self.stats.discussion_times.append(datetime.now(timezone.utc))

        discussion = await self.orchestrator.discuss(event=event)
        proposal = discussion.proposal

        logger.info(
            "proposal_generated",
            event_id=event.id,
            ticker=proposal.ticker,
            conviction=proposal.conviction,
            direction=proposal.direction,
        )

        ragas = await self.evaluator.evaluate(proposal, context_limit=10)
        proposal = proposal.model_copy(update={"ragas_score": ragas.ragas_score})
        self.repository.save_trade_proposal(proposal)

        min_required_ragas = max(
            float(self.settings.ragas_min_score),
            float(self.risk_config.min_ragas_score),
        )
        if ragas.ragas_score < min_required_ragas:
            self.stats.proposals_rejected += 1
            logger.info(
                "proposal_rejected_ragas",
                proposal_id=proposal.id,
                ragas_score=ragas.ragas_score,
                threshold=min_required_ragas,
            )
            return

        execution_result = await self.executor.execute_proposal(proposal, current_prices={})
        if execution_result.success:
            self.stats.trades_placed += 1
            logger.info(
                "trade_placed",
                ticker=execution_result.ticker,
                order_id=execution_result.order_id,
            )
            return

        self.stats.proposals_rejected += 1
        logger.info(
            "proposal_rejected_execution",
            ticker=execution_result.ticker,
            reason=execution_result.error,
            risk_checks=execution_result.risk_checks,
        )

    async def _run_all_scanners_once(self) -> None:
        """Run each enabled scanner once and queue produced events."""
        for source in self.bundle.sources.sources:
            if not source.enabled:
                continue

            scanner = build_scanner(
                source=source,
                settings=self.settings,
                repository=self.repository,
                fact_store=self.fact_store,
                sentiment=self.sentiment,
                ner=self.ner,
                scorer=self.scorer,
            )
            if scanner is None:
                continue

            try:
                queued_events = await scanner.run()
                self.stats.events_scanned += len(queued_events)
                for event in queued_events:
                    if self.event_queue.full():
                        logger.warning("event_queue_full_drop", source=source.name, event_id=event.id)
                        continue
                    await self.event_queue.put(event)
                logger.info(
                    "scanner_cycle_complete_once",
                    source=source.name,
                    queued=len(queued_events),
                    queue_size=self.event_queue.qsize(),
                )
            except Exception as exc:
                logger.exception("scanner_once_error", source=source.name, error=str(exc))

    async def _enqueue_top_recent_events_for_forced_discussion(self, limit: int) -> None:
        """Queue top recent events by importance, ignoring dedup/newness checks."""
        if limit <= 0:
            return

        recent = self.repository.fetch_recent_events(limit=max(200, limit * 10))
        if not recent:
            logger.info("force_discuss_no_recent_events")
            return

        recent.sort(key=lambda event: event.importance_score, reverse=True)
        selected = recent[:limit]

        enqueued = 0
        for event in selected:
            if self.event_queue.full():
                logger.warning("force_discuss_queue_full_drop", event_id=event.id)
                continue
            await self.event_queue.put(event)
            enqueued += 1

        logger.info(
            "force_discuss_events_enqueued",
            requested=limit,
            enqueued=enqueued,
            candidate_pool=len(recent),
            queue_size=self.event_queue.qsize(),
        )

    async def _initialize_first_run(self) -> None:
        """Warm start scanners: persist current source state without queueing events."""
        logger.info("first_run_initialization_start")

        for source in self.bundle.sources.sources:
            if not source.enabled:
                continue

            scanner = build_scanner(
                source=source,
                settings=self.settings,
                repository=self.repository,
                fact_store=self.fact_store,
                sentiment=self.sentiment,
                ner=self.ner,
                scorer=self.scorer,
            )
            if scanner is None:
                continue

            try:
                await scanner.run(warmup=True)
            except Exception as exc:
                logger.exception("first_run_warmup_error", source=source.name, error=str(exc))

        self.first_run_initialized = True
        logger.info("first_run_initialization_complete", action="baseline_persisted_no_queue")

    async def _drain_event_queue_once(self) -> None:
        """Process up to 3 events concurrently using asyncio.Semaphore, sorted by importance."""
        # Collect all events from queue
        events_to_process: list[tuple] = []
        while not self.event_queue.empty():
            event = await self.event_queue.get()
            events_to_process.append((event.importance_score, event))
            self.event_queue.task_done()
        
        # Sort by importance_score descending
        events_to_process.sort(key=lambda x: x[0], reverse=True)
        max_size = self.settings.max_queue_size
        
        # Limit concurrency to 3 events with Semaphore
        semaphore = asyncio.Semaphore(3)
        
        async def process_with_semaphore(importance: float, event) -> None:
            async with semaphore:
                try:
                    await self._process_event(event)
                except Exception as exc:
                    logger.exception("event_once_error", error=str(exc))
        
        # Process top max_queue_size events concurrently (up to 3 at a time)
        tasks = []
        for idx, (importance, event) in enumerate(events_to_process):
            if idx < max_size:
                tasks.append(process_with_semaphore(importance, event))
            else:
                # Put remaining events back in queue for next cycle
                await self.event_queue.put(event)
        
        # Run tasks concurrently
        if tasks:
            await asyncio.gather(*tasks)

    async def _position_monitor_loop(self) -> None:
        """Monitor brackets and open positions on a fixed cadence."""
        while not self.shutdown_event.is_set():
            try:
                await self.executor.monitor_and_update_positions()
            except Exception as exc:
                logger.warning("position_monitor_error", error=str(exc))

            try:
                await asyncio.wait_for(self.shutdown_event.wait(), timeout=30)
            except asyncio.TimeoutError:
                pass

    async def _status_loop(self) -> None:
        """Print operational counters every 5 minutes."""
        while not self.shutdown_event.is_set():
            now = datetime.now(timezone.utc)
            self.stats.last_status_at = now
            print(
                "[STATUS] "
                f"{now.isoformat()} "
                f"events_scanned={self.stats.events_scanned} "
                f"discussions_triggered={self.stats.discussions_triggered} "
                f"trades_placed={self.stats.trades_placed} "
                f"queue_size={self.event_queue.qsize()}"
            )

            try:
                await asyncio.wait_for(self.shutdown_event.wait(), timeout=300)
            except asyncio.TimeoutError:
                pass

    def _can_trigger_discussion_now(self) -> bool:
        """Enforce discussions_per_day from settings.yaml."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=1)
        self.stats.discussion_times = [ts for ts in self.stats.discussion_times if ts > cutoff]
        return len(self.stats.discussion_times) < int(self.settings.discussions_per_day)

    async def _shutdown(self) -> None:
        """Stop tasks, drain resources, and disconnect broker."""
        for task in self.tasks:
            task.cancel()

        for task in self.tasks:
            with suppress(asyncio.CancelledError):
                await task

        with suppress(Exception):
            await self.ibkr.disconnect()

        logger.info(
            "autonomous_loop_stopped",
            events_scanned=self.stats.events_scanned,
            discussions_triggered=self.stats.discussions_triggered,
            trades_placed=self.stats.trades_placed,
            proposals_rejected=self.stats.proposals_rejected,
        )


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Defense trading autonomous loop")
    parser.add_argument("--once", action="store_true", help="Run one full scan->discuss->evaluate->execute cycle and exit")
    parser.add_argument(
        "--force-discuss",
        action="store_true",
        help="Testing mode: skip first-run warmup and replay top recent events through discuss/evaluate/execute.",
    )
    args = parser.parse_args()

    bundle = load_config_bundle("config")
    configure_logging(bundle.settings)

    app = AutonomousTradingApp(bundle)
    try:
        if args.once:
            asyncio.run(app.run_once(force_discuss=args.force_discuss))
        else:
            asyncio.run(app.run())
    except KeyboardInterrupt:
        # Fallback for environments where signal handlers are restricted.
        print("\nKeyboardInterrupt received. Exiting.")


if __name__ == "__main__":
    main()
