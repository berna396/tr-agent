import logging
from datetime import datetime

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from tr_agent import journal, notifier
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
    journal.init()

    broker = PaperBroker(
        initial_capital=settings.paper_initial_capital,
        slippage=settings.paper_slippage,
    )
    orders_placed = []

    log.info(f"[Cycle] Starting — {len(tickers)} tickers: {', '.join(tickers)}")

    for ticker in tickers:
        log.info(f"[Cycle] Analyzing {ticker}...")

        # Stage 1: Signal detection
        try:
            analysis = technical.analyze(ticker)
        except Exception as e:
            log.error(f"[Cycle] Failed to analyze {ticker}: {e}")
            continue

        journal.log_signal(
            ticker=ticker, signal=analysis.signal.value,
            rsi=analysis.rsi, macd_hist=analysis.macd_hist,
            sma_20=analysis.sma_20, sma_50=analysis.sma_50,
            close=analysis.close, reasoning=analysis.reasoning,
        )
        log.info(f"[Cycle] {ticker}: {analysis.signal.upper()} — {analysis.reasoning}")

        if analysis.signal == Signal.NEUTRAL:
            log.info(f"[Cycle] {ticker}: no signal, skipping")
            continue

        # Stage 2: Risk check
        portfolio = broker.get_portfolio()
        quote = broker.get_quote(ticker)
        risk_check = risk.evaluate(analysis, portfolio, quote)

        journal.log_risk(ticker, risk_check.approved, risk_check.reason, risk_check.max_quantity)

        if not risk_check.approved:
            log.info(f"[Cycle] {ticker}: risk rejected — {risk_check.reason}")
            notifier.send(f"⚠️ *{ticker}* signal blocked by risk manager\n_{risk_check.reason}_")
            continue

        # Stage 3: LLM confirmation (with memory context)
        decision = llm.confirm_trade(analysis, risk_check, portfolio)
        journal.log_llm_decision(ticker, decision.confirmed, decision.quantity, decision.reasoning)
        log.info(f"[Cycle] {ticker}: LLM {'confirmed' if decision.confirmed else 'rejected'} — {decision.reasoning}")

        if not decision.confirmed or decision.quantity <= 0:
            continue

        # Stage 4: Execute
        quantity = min(decision.quantity, risk_check.max_quantity)
        try:
            order = broker.place_order(ticker, decision.side, quantity)
            journal.log_order(ticker, order.side.value, order.quantity, order.fill_price, order.order_id)
            log.info(f"[Cycle] {ticker}: order placed — {order.side.value} {quantity:.4f} @ ${order.fill_price:.2f}")

            # If this was a SELL, record the outcome against the pending BUY
            if order.side.value == "sell":
                pending = journal.get_pending_buy(ticker)
                if pending:
                    journal.record_outcome(
                        ticker=ticker,
                        buy_ts=pending["ts"],
                        sell_ts=datetime.now(ET).isoformat(),
                        buy_price=pending["fill_price"],
                        sell_price=order.fill_price,
                        quantity=order.quantity,
                        buy_reasoning=analysis.reasoning,
                        sell_reasoning=decision.reasoning,
                    )

            orders_placed.append({
                "ticker": order.ticker,
                "side": order.side.value,
                "quantity": order.quantity,
                "fill_price": order.fill_price,
            })
        except Exception as e:
            log.error(f"[Cycle] {ticker}: order failed — {e}")

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

    scheduler.add_job(
        run_cycle,
        CronTrigger(
            day_of_week="mon-fri",
            hour="9-15",
            minute="30,0",
            timezone=ET,
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
