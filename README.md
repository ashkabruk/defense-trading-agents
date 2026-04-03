# defense-trading-agents

Multi-agent system that scans US defense sector data, debates trade ideas through specialized AI agents, fact-checks proposals, and executes on IBKR paper trading.

## What it does

Nine scanners monitor public sources (SAM.gov contracts, SEC filings, FRED macro data, defense news, congressional trades, etc.) for signals relevant to defense stocks. When something interesting comes in, five AI agents - a contracts analyst, macro economist, sentiment analyst, devil's advocate, and risk manager - discuss it across multiple rounds. A RAGAS-inspired evaluator then checks whether the agents' claims are actually grounded in retrieved evidence. If the proposal passes both the evaluation and hard-coded risk limits, a bracket order (entry + stop-loss + take-profit) is placed on IBKR.

The system rejected its first proposal (LMT sustainment contract, conviction 4.4/10 - agents disagreed, risk manager vetoed) and executed its second (NOC ICBM contract, conviction 8.2/10, RAGAS 0.847, 93 shares @ $702.75).

## Architecture

```
Scanners (9 sources, async)
    → FinBERT sentiment + spaCy NER (local, no API)
    → SQLite dedup + ChromaDB embeddings
    → Agent discussion (5 agents × up to 3 rounds, parallel within rounds)
    → RAGAS evaluation (claim decomposition + fact grounding, parallel scoring)
    → Risk manager (hard limits, correlation check, earnings blackout)
    → IBKR bracket order (limit entry + trailing stop + take-profit)
```

## Running it

```bash
# one full cycle (scan → discuss → evaluate → execute → exit)
python src/main.py --once

# continuous autonomous mode
python src/main.py

# test with existing data (forces discussion on stored events)
python src/main.py --once --force-discuss
```

Requires `.env` with API keys:
```
DEEPSEEK_API_KEY=...
GEMINI_API_KEY=...
SAM_API_KEY=...
FRED_API_KEY=...
```

IB Gateway must be running on `localhost:4002` (paper trading) for order execution.

## Config

Everything is controlled via YAML in `config/`:

- `settings.yaml` - model endpoints, scan intervals, thresholds, queue size
- `agents.yaml` - agent prompts, roles, tools, speaking order (add/remove agents here)
- `sources.yaml` - data source URLs, parsers, polling intervals, keyword filters
- `risk.yaml` - position limits, allowed tickers, daily loss cap, IBKR connection

Swapping the discussion model from DeepSeek to Claude is a one-line config change.

## Stack

Python 3.12, asyncio, DeepSeek V3.2 (via OpenAI SDK), Gemini 2.5 Flash-Lite, FinBERT, ib_insync, SQLite, ChromaDB, Pydantic v2, structlog.

Runs at ~$12/month (DeepSeek API + small VPS).

## Project structure

```
src/
├── main.py                  # entry point, autonomous loop
├── llm/client.py            # unified LLM client (DeepSeek, Gemini, Anthropic)
├── agents/
│   ├── agent.py             # single agent turn logic
│   ├── orchestrator.py      # multi-round discussion with early consensus detection
│   └── tools.py             # tool registry (fact search, FRED lookup, materiality calc)
├── evaluation/ragas.py      # claim decomposition + faithfulness/relevance scoring
├── execution/
│   ├── ibkr.py              # IBKR connector with auto-reconnect
│   ├── risk.py              # risk limits, correlation matrix, earnings blackout
│   ├── orders.py            # bracket orders, trailing stops, fill monitoring
│   └── executor.py          # coordinates risk → orders → broker
├── scanners/                # one module per data source
├── processing/              # FinBERT, spaCy NER, importance scoring
├── storage/                 # SQLite + ChromaDB
└── models/                  # Pydantic models
```

## Scanners

| Source | Method | Interval |
|--------|--------|----------|
| SAM.gov | REST API | 60 min |
| defense.gov (war.gov) | RSS | 60s |
| Truth Social | RSS | 30s |
| FRED | REST API | 60 min |
| SEC EDGAR | full-text search | 15 min |
| Capitol Trades | scraper | 60 min |
| Google News | RSS | 60s |
| GAO | RSS | 60 min |
| DSCA | scraper | disabled |

## TODO

- [ ] LLM-powered contradiction detection in RAGAS (currently keyword heuristic)
- [ ] trade journal with full reasoning chain for backtesting
- [ ] weekly P&L report with Sharpe ratio and max drawdown
- [ ] feedback loop: store trade outcomes in ChromaDB so agents learn from past trades
- [ ] Telegram/ntfy.sh alerts for trades and daily summary