import logging
from datetime import datetime

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from tr_agent import notifier
from tr_agent.agent import core as llm
from tr_agent.broker.paper import PaperBroker
from tr_agent.config import DEFAULT_WATCHLIST, settings
from tr_agent.signals import technical
from tr_agent.signals.technical import Signal
from tr_agent import risk

log = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")


def run_cycle(tickers: list[str] | None = None) -> None:
    tickers = tickers or DEFAULT_WATCHLIST
    broker = PaperBroker(
        initial_capital=settings.paper_initial_capital,
        slippage=settings.paper_slippage,
    )
    orders_placed = []

    log.info(f"[Cycle] Starting — {len(tickers)} tickers: {', '.join(tickers)}")

    for ticker in tickers:
        log.info(f"[Cycle] Analyzing {ticker}...")

        # Stage 1: Signal detection (pure Python, no LLM)
        try:
            analysis = technical.analyze(ticker)
        except Exception as e:
            log.error(f"[Cycle] Failed to analyze {ticker}: {e}")
            continue

        log.info(f"[Cycle] {ticker}: {analysis.signal.upper()} — {analysis.reasoning}")

        if analysis.signal == Signal.NEUTRAL:
            log.info(f"[Cycle] {ticker}: no signal, skipping")
            continue

        # Stage 2: Risk check (pure Python, no LLM)
        portfolio = broker.get_portfolio()
        quote = broker.get_quote(ticker)
        risk_check = risk.evaluate(analysis, portfolio, quote)

        if not risk_check.approved:
            log.info(f"[Cycle] {ticker}: risk rejected — {risk_check.reason}")
            notifier.send(f"⚠️ *{ticker}* signal blocked by risk manager\n_{risk_check.reason}_")
            continue

        # Stage 3: LLM confirmation
        decision = llm.confirm_trade(analysis, risk_check, portfolio)
        log.info(f"[Cycle] {ticker}: LLM {'confirmed' if decision.confirmed else 'rejected'} — {decision.reasoning}")

        if not decision.confirmed or decision.quantity <= 0:
            log.info(f"[Cycle] {ticker}: LLM rejected trade")
            continue

        # Stage 4: Execute
        quantity = min(decision.quantity, risk_check.max_quantity)
        try:
            order = broker.place_order(ticker, decision.side, quantity)
            log.info(f"[Cycle] {ticker}: order placed — {order.side.value} {quantity:.4f} @ ${order.fill_price:.2f}")
            orders_placed.append({
                "ticker": order.ticker,
                "side": order.side.value,
                "quantity": order.quantity,
                "fill_price": order.fill_price,
            })
        except Exception as e:
            log.error(f"[Cycle] {ticker}: order failed — {e}")

    # Send Telegram summary
    metrics = broker.get_metrics(tickers)
    notifier.send_run_summary(tickers, "", metrics, orders_placed)
    log.info(f"[Cycle] Done — {len(orders_placed)} trades executed")


def start(tickers: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    scheduler = BlockingScheduler(timezone=ET)

    # Run every 30 min Mon-Fri during NYSE market hours (9:30 AM – 4:00 PM ET)
    scheduler.add_job(
        run_cycle,
        CronTrigger(
            day_of_week="mon-fri",
            hour="9-15",
            minute="30,0",
            timezone=ET,
            # Skip the 9:00 AM slot (market opens at 9:30)
            start_date=datetime.now(ET).replace(hour=9, minute=30, second=0),
        ),
        kwargs={"tickers": tickers},
        id="trading_cycle",
        name="Trading cycle",
        misfire_grace_time=300,
    )

    log.info(f"Scheduler started — running every 30 min during NYSE hours (9:30–16:00 ET)")
    log.info(f"Watchlist: {', '.join(tickers or DEFAULT_WATCHLIST)}")
    log.info("Press Ctrl+C to stop")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        log.info("Scheduler stopped")
