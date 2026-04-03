from __future__ import annotations

import structlog

from src.models.config import Settings, SourceConfig
from src.processing.ner import NERProcessor
from src.processing.scoring import ImportanceScorer
from src.processing.sentiment import SentimentAnalyzer
from src.scanners.base import BaseScanner
from src.scanners.capitol_trades import CapitolTradesScanner
from src.scanners.defense_gov_rss import DefenseGovRssScanner
from src.scanners.dsca import DscaScanner
from src.scanners.fred import FredScanner
from src.scanners.gao import GaoScanner
from src.scanners.google_news import GoogleNewsScanner
from src.scanners.sam_gov import SamGovScanner
from src.scanners.sec_edgar import SecEdgarScanner
from src.scanners.truth_social import TruthSocialScanner
from src.storage.db import SQLiteRepository
from src.storage.fact_store import ChromaFactStore

logger = structlog.get_logger(__name__)

SCANNER_BY_PARSER: dict[str, type[BaseScanner]] = {
    "sam_gov": SamGovScanner,
    "defense_gov_rss": DefenseGovRssScanner,
    "truth_social": TruthSocialScanner,
    "google_news": GoogleNewsScanner,
    "fred": FredScanner,
    "sec_edgar": SecEdgarScanner,
    "capitol_trades": CapitolTradesScanner,
    "gao": GaoScanner,
    "dsca": DscaScanner,
}


def build_scanner(
    source: SourceConfig,
    settings: Settings,
    repository: SQLiteRepository,
    fact_store: ChromaFactStore,
    sentiment: SentimentAnalyzer,
    ner: NERProcessor,
    scorer: ImportanceScorer,
) -> BaseScanner | None:
    """Create scanner instance from parser key in config."""
    scanner_cls = SCANNER_BY_PARSER.get(source.parser)
    if scanner_cls is None:
        logger.warning("scanner_not_implemented", source=source.name, parser=source.parser)
        return None

    return scanner_cls(
        source=source,
        settings=settings,
        repository=repository,
        fact_store=fact_store,
        sentiment=sentiment,
        ner=ner,
        scorer=scorer,
    )
