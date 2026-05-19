import logging
from datetime import datetime

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from pathlib import Path

from tr_agent import guards, journal, market_regime, notifier, risk, screener
from tr_agent.agent import core as llm
from tr_agent.agent.news_analyst import analyze_news
from tr_agent.broker.base import OrderSide
from tr_agent.broker.paper import PaperBroker
from tr_agent.config import DEFAULT_WATCHLIST, settings
from tr_agent.ml.auto_improve import daily_ml_check, weekly_analysis
from tr_agent.signals import technical
from tr_agent.signals.technical import Signal

_WATCHLIST_PATH = Path(__file__).parents[2] / "data" / "active_watchlist.json"

log = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")


def refresh_watchlist(tickers: list[str] | None = None) -> None:
    """Run the pre-market screener and update active_watchlist.json."""
    log.info("[Screener] Running pre-market scan...")
    pool = tickers or None  # None → uses CANDIDATE_POOL
    selected = screener.screen(pool=pool)
    screener.save_active_watchlist(selected, _WATCHLIST_PATH)
    log.info(f"[Screener] Watchlist updated: {', '.join(selected)}")


def run_cycle(tickers: list[str] | None = None, regime_ticker: str = "SPY") -> None:
    if tickers is None:
        tickers = screener.load_active_watchlist(_WATCHLIST_PATH)
    else:
        tickers = tickers or DEFAULT_WATCHLIST
    journal.init()

    broker = PaperBroker(
        initial_capital=settings.paper_initial_capital,
        slippage=settings.paper_slippage,
    )
    orders_placed = []
    stop_loss_sold: set[str] = set()

    log.info(f"[Cycle] Starting — {len(tickers)} tickers: {', '.join(tickers)}")

    # Stage 0: Stop-loss enforcement — runs before signal detection
    portfolio = broker.get_portfolio()
    for sl_ticker, position in list(portfolio.positions.items()):
        try:
            quote = broker.get_quote(sl_ticker)
            loss_pct = (quote.price - position.avg_price) / position.avg_price

            # ATR-based stop if stored on position; otherwise fall back to fixed %
            if position.stop_price is not None:
                triggered = quote.price <= position.stop_price
                threshold_str = f"stop ${position.stop_price:.2f} (ATR-based)"
            elif settings.stop_loss_pct > 0:
                triggered = loss_pct <= -settings.stop_loss_pct
                threshold_str = f"fixed {settings.stop_loss_pct:.0%}"
            else:
                triggered = False
                threshold_str = ""

            if triggered:
                order = broker.place_order(sl_ticker, OrderSide.SELL, position.quantity)
                journal.log_order(
                    sl_ticker, "sell", order.quantity, order.fill_price, order.order_id
                )
                pending = journal.get_pending_buy(sl_ticker)
                if pending:
                    journal.record_outcome(
                        ticker=sl_ticker,
                        buy_ts=pending["ts"],
                        sell_ts=datetime.now(ET).isoformat(),
                        buy_price=pending["fill_price"],
                        sell_price=order.fill_price,
                        quantity=order.quantity,
                        buy_reasoning="",
                        sell_reasoning=f"stop-loss at {loss_pct:.1%} ({threshold_str})",
                    )
                stop_loss_sold.add(sl_ticker)
                orders_placed.append({
                    "ticker": sl_ticker, "side": "sell",
                    "quantity": order.quantity, "fill_price": order.fill_price,
                })
                log.warning(
                    f"[StopLoss] {sl_ticker}: sold {order.quantity:.4f} @ "
                    f"${order.fill_price:.2f} ({loss_pct:+.1%}, {threshold_str})"
                )
                notifier.send_trade_slack(
                    ticker=sl_ticker,
                    side="sell",
                    quantity=order.quantity,
                    price=order.fill_price,
                    pnl_pct=loss_pct * 100,
                    reason="stop-loss",
                )
        except Exception as e:
            log.error(f"[StopLoss] {sl_ticker}: {e}")

    # Get market regime once per cycle
    regime = market_regime.get_regime(regime_ticker)

    for ticker in tickers:
        if ticker in stop_loss_sold:
            log.info(f"[Cycle] {ticker}: skipping — stop-loss triggered this cycle")
            continue

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
            ml_features=analysis.ml_features or None,
        )
        log.info(f"[Cycle] {ticker}: {analysis.signal.upper()} — {analysis.reasoning}")

        if analysis.signal == Signal.NEUTRAL:
            continue

        # BUY-only guards: regime filter and earnings blackout
        earnings_days: int | None = None
        if analysis.signal == Signal.BUY:
            if settings.regime_filter_enabled and not regime.bullish:
                log.info(
                    f"[Cycle] {ticker}: BUY suppressed — {regime.label} regime "
                    f"(SPY SMA50={regime.sma50:.0f} < SMA200={regime.sma200:.0f})"
                )
                continue

            earnings_days = guards.days_until_earnings(ticker)
            if settings.earnings_blackout_days > 0 and guards.is_earnings_blackout(
                ticker, days_before=settings.earnings_blackout_days
            ):
                log.info(f"[Cycle] {ticker}: BUY suppressed — earnings blackout")
                continue

        # Stage 1.5: News analysis — structured LLM context replacing raw headlines
        news_ctx = analyze_news(ticker, analysis.headlines, earnings_days_away=earnings_days)
        if analysis.signal == Signal.BUY and settings.news_risk_gate:
            if news_ctx.is_high_risk and news_ctx.sentiment_score < -0.5:
                log.info(
                    f"[Cycle] {ticker}: BUY suppressed — news risk gate "
                    f"(risk={news_ctx.risk_level} sentiment={news_ctx.sentiment_score:.2f})"
                )
                continue

        # Stage 2: Risk check
        portfolio = broker.get_portfolio()
        quote = broker.get_quote(ticker)
        risk_check = risk.evaluate(analysis, portfolio, quote)

        journal.log_risk(ticker, risk_check.approved, risk_check.reason, risk_check.max_quantity)

        if not risk_check.approved:
            log.info(f"[Cycle] {ticker}: risk rejected — {risk_check.reason}")
            continue

        # Stage 3: LLM confirmation (with memory context, regime, and news context)
        decision = llm.confirm_trade(analysis, risk_check, portfolio, regime=regime, news_ctx=news_ctx)
        journal.log_llm_decision(ticker, decision.confirmed, decision.quantity, decision.reasoning)
        log.info(
            f"[Cycle] {ticker}: LLM {'confirmed' if decision.confirmed else 'rejected'} "
            f"— {decision.reasoning}"
        )

        if not decision.confirmed or decision.quantity <= 0:
            continue

        # Stage 4: Execute
        quantity = min(decision.quantity, risk_check.max_quantity)
        try:
            atr_stop: float | None = None
            if decision.side == OrderSide.BUY:
                atr_ratio = analysis.ml_features.get("atr_ratio")
                if atr_ratio and analysis.close:
                    atr_stop = round(
                        analysis.close - settings.stop_loss_atr_multiplier * atr_ratio * analysis.close,
                        4,
                    )
            order = broker.place_order(ticker, decision.side, quantity, stop_price=atr_stop)
            journal.log_order(ticker, order.side.value, order.quantity, order.fill_price, order.order_id)
            stop_str = f" | stop ${atr_stop:.2f}" if atr_stop else ""
            log.info(
                f"[Cycle] {ticker}: order placed — {order.side.value} "
                f"{quantity:.4f} @ ${order.fill_price:.2f}{stop_str}"
            )

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

            cash_pct = risk_check.max_quantity * order.fill_price / (portfolio.cash or 1)
            pnl_pct_val: float | None = None
            if order.side.value == "sell":
                pending = journal.get_pending_buy(ticker)
                if pending:
                    pnl_pct_val = (order.fill_price - pending["fill_price"]) / pending["fill_price"] * 100
            notifier.send_trade_slack(
                ticker=ticker,
                side=order.side.value,
                quantity=order.quantity,
                price=order.fill_price,
                cash_pct=cash_pct if order.side.value == "buy" else None,
                stop_price=atr_stop if order.side.value == "buy" else None,
                pnl_pct=pnl_pct_val,
            )
        except Exception as e:
            log.error(f"[Cycle] {ticker}: order failed — {e}")

    metrics = broker.get_metrics(tickers)
    log.info(
        f"[Cycle] Done — {len(orders_placed)} trades executed | "
        f"portfolio ${metrics['total_value']:,.2f} ({metrics['total_return_pct']:+.2f}%)"
    )


def run_crypto_cycle() -> None:
    """Trading cycle for crypto assets — runs 24/7, uses BTC regime."""
    tickers = settings.crypto_watchlist
    if not tickers:
        return
    run_cycle(tickers=tickers, regime_ticker=settings.crypto_regime_ticker)


def start(tickers: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    scheduler = BlockingScheduler(timezone=ET)

    scheduler.add_job(
        refresh_watchlist,
        CronTrigger(day_of_week="mon-fri", hour=9, minute=15, timezone=ET),
        id="screener_daily",
        name="Pre-market watchlist screener",
        misfire_grace_time=600,
    )

    scheduler.add_job(
        run_cycle,
        CronTrigger(
            day_of_week="mon-fri",
            hour="9-15",
            minute="30,0",
            timezone=ET,
            start_date=datetime.now(ET).replace(hour=9, minute=30, second=0),
        ),
        # No tickers kwarg — run_cycle reads active_watchlist.json on every call,
        # which the 9:15 screener updates each morning.
        id="trading_cycle",
        name="Trading cycle",
        misfire_grace_time=300,
    )

    scheduler.add_job(
        daily_ml_check,
        CronTrigger(day_of_week="mon-fri", hour=9, minute=0, timezone=ET),
        kwargs={"tickers": tickers},
        id="ml_daily_check",
        name="ML daily retrain check",
        misfire_grace_time=600,
    )

    scheduler.add_job(
        weekly_analysis,
        CronTrigger(day_of_week="sun", hour=10, minute=0, timezone=ET),
        kwargs={"tickers": tickers},
        id="ml_weekly_analysis",
        name="ML weekly analysis",
        misfire_grace_time=3600,
    )

    if settings.crypto_watchlist:
        scheduler.add_job(
            run_crypto_cycle,
            CronTrigger(minute="0,30"),  # every 30 min, 24/7
            id="crypto_cycle",
            name="Crypto trading cycle (24/7)",
            misfire_grace_time=300,
        )

    active = screener.load_active_watchlist(_WATCHLIST_PATH)
    log.info(f"Scheduler started — running every 30 min during NYSE hours (9:30–16:00 ET)")
    log.info(f"Active watchlist ({len(active)}): {', '.join(active)}")
    log.info("Press Ctrl+C to stop")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        log.info("Scheduler stopped")
