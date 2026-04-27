from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional
from math import ceil, erf, floor, log, sqrt
import json


# ─────────────────────────────────────────────────────────────
# MARK COUNTERPARTY MODULE (v6 — defensive-only bias on HG and VE)
# ─────────────────────────────────────────────────────────────

MARK_EDGE_VE: Dict[str, float] = {
    "Mark 01": +2.94,
    "Mark 14": +1.92,
    "Mark 22": +0.33,
    "Mark 55": -1.43,
    "Mark 67": +1.25,
}

# Hydrogel: Mark 14 is +7.5 (very strongly informed), Mark 38 is -7.1
# (strongly anti-correlated → noise / contrarian).  Both have meaningful
# absolute edges so both should influence our take edges.
MARK_EDGE_HG: Dict[str, float] = {
    "Mark 14": +7.50,
    "Mark 38": -7.13,
}

# Persistent flow parameters (shared between products)
MARK_FLOW_PERSISTENCE_TICKS = 400
MARK_FLOW_KEY_VE = "ve_mark_flow"
MARK_FLOW_TS_KEY_VE = "ve_mark_flow_ts"
MARK_FLOW_KEY_HG = "hg_mark_flow"
MARK_FLOW_TS_KEY_HG = "hg_mark_flow_ts"

# Defensive take-edge widening.  Asymmetric: when flow > 0 (informed
# buying), only widen the SELL threshold (refuse to sell to them).
# When flow < 0 (informed selling), only widen the BUY threshold
# (refuse to buy from them).  Never narrow either threshold.
# v8: VE bias dropped to 0 (v7 confirmed this recovers $1,333).
# HG bias kept active — HG signal is much stronger and works.
VE_DEFENSIVE_BIAS = 0.0
HG_DEFENSIVE_BIAS = 12.0


def _instant_flow(state: TradingState, product: str,
                  edge_table: Dict[str, float]) -> Optional[float]:
    """Snapshot Mark flow from current market_trades, or None if no
    informative trades visible this tick."""
    market_trades = (state.market_trades or {}).get(product, [])
    if not market_trades:
        return None
    flow = 0.0
    weight_total = 0.0
    for t in market_trades:
        buyer = getattr(t, "buyer", None) or ""
        seller = getattr(t, "seller", None) or ""
        if buyer == "SUBMISSION" or seller == "SUBMISSION":
            continue
        if not buyer and not seller:
            continue
        qty = abs(int(getattr(t, "quantity", 1) or 1))
        b_edge = edge_table.get(buyer, 0.0)
        s_edge = edge_table.get(seller, 0.0)
        contrib = (b_edge - s_edge) * qty
        flow += contrib
        weight_total += qty
    if weight_total <= 0:
        return None
    raw = flow / weight_total
    return max(-1.0, min(1.0, raw))


def _persistent_flow(state: TradingState, trader_data: Dict, product: str,
                     edge_table: Dict[str, float],
                     key_flow: str, key_ts: str) -> float:
    """Persistent informed-flow signal in roughly [-1, +1] for given
    product, using exponential decay through traderData."""
    now = state.timestamp
    cached = float(trader_data.get(key_flow, 0.0))
    last_ts = int(trader_data.get(key_ts, now))
    age = max(0, now - last_ts)
    decay = 0.5 ** (age / MARK_FLOW_PERSISTENCE_TICKS) if age > 0 else 1.0
    decayed = cached * decay

    inst = _instant_flow(state, product, edge_table)
    if inst is not None:
        new_flow = (inst + decayed * decay) / (1.0 + decay)
        new_flow = max(-1.0, min(1.0, new_flow))
        trader_data[key_flow] = new_flow
        trader_data[key_ts] = now
        return new_flow
    return decayed


def compute_ve_flow(state: TradingState, trader_data: Dict) -> float:
    return _persistent_flow(
        state, trader_data, "VELVETFRUIT_EXTRACT", MARK_EDGE_VE,
        MARK_FLOW_KEY_VE, MARK_FLOW_TS_KEY_VE,
    )


def compute_hg_flow(state: TradingState, trader_data: Dict) -> float:
    return _persistent_flow(
        state, trader_data, "HYDROGEL_PACK", MARK_EDGE_HG,
        MARK_FLOW_KEY_HG, MARK_FLOW_TS_KEY_HG,
    )





POSITION_LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    "VEV_4000": 300,
    "VEV_4500": 300,
    "VEV_5000": 300,
    "VEV_5100": 300,
    "VEV_5200": 300,
    "VEV_5300": 300,
    "VEV_5400": 300,
    "VEV_5500": 300,
    "VEV_6000": 300,
    "VEV_6500": 300,
}

TRADE_UNDERLYING = True
TRADE_OPTIONS = True
TRADE_HYDROGEL = True

UNDERLYING = "VELVETFRUIT_EXTRACT"

# All strikes used to FIT the IV smile (more points = stabler curvature)
ACTIVE_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]

# Strikes traded as DELTA-ONE instruments (intrinsic ≈ mid, BS pricing
# is meaningless because time value is ~0).  These get the same VE-style
# market-making treatment with fair = spot − K, and their positions feed
# the existing VE delta hedge automatically.
#   - VEV_4000: spot ∈ [5191, 5300] → always 1191–1300 ITM, time value
#     in [-7, +7] around zero, 442 trades over 3 days, ~22-wide market
#     spread around intrinsic.  Plenty of edge to capture.
#   - VEV_4500: same pricing structure but only 3 trades in 3 days —
#     dead market, not worth quoting.
DELTA_ONE_STRIKES = {4000}

# Strikes we actually QUOTE on as options (smile fit + sell premium).
# Round-4 analysis (3 days):
#   - 4000:    deep ITM, ~$0 time value → DELTA-ONE (handled separately)
#   - 4500:    deep ITM, dead market → drop
#   - 5000/5100: ITM, thin Mark flow, lost money last run → drop
#   - 5200/5300: ATM-ish, $41 avg time value, Mark 22 sells size → KEEP
#   - 5400/5500: OTM, Mark 22 sells huge size → KEEP
#   - 6000/6500: far OTM, pinned 0/1 book, post asks for free fills → KEEP
TRADE_STRIKES = [5200, 5300, 5400, 5500, 6000, 6500]


# ── HYDROGEL_PACK (unchanged) ─────────────────────────────

HG_FAIR = 9991
HG_TAKE_EDGE = 26
HG_TAKE_SIZE = 20
HG_POST_EDGE = 8
HG_POST_SIZE = 20
HG_SKEW_DIV = 25

# v6: blended anchor.  Original anchor of $9,991 + small momentum
# left fair pinned ~$10,000 even when the market traded at $10,060.
# Use a SLOW EMA (alpha=0.005) as a drifting anchor and blend with
# the original HG_FAIR.  ANCHOR_WEIGHT = 0.4 means 40% original /
# 60% slow EMA.
HG_SLOW_EMA_ALPHA = 0.005
HG_SLOW_KEY = "HYDROGEL_PACK_slow_ema"
HG_ANCHOR_WEIGHT = 0.4   # 40% to fixed $9,991 anchor (500244 value, ~$5,937 HG vs $4,212 at 0.3)

# ── HYDROGEL momentum overlay ─────────────────────────────
HG_MOM_ALPHA = 0.12
HG_MOM_K = 0.45
HG_MOM_CAP = 6
HG_MOM_TAKE_BOOST = 1.3
HG_MOM_KEY = "HYDROGEL_PACK_ema"


# ── VELVETFRUIT_EXTRACT (unchanged) ───────────────────────

VE_ANCHOR = 5262.0
VE_ANCHOR_WEIGHT = 0.6
VE_EMA_ALPHA = 0.08
VE_IMBALANCE_K = 2.0
VE_IMBALANCE_CAP = 2.0
VE_SKEW_DIV = 60

VE_EDGE = 7
VE_SIZE = 20
VE_POST_EDGE = 1
VE_POST_SIZE = 20

# v14: Partial hedging.  Full delta-hedging (1.0) pinned VE to its +200
# cap last run and cost ~$4k of VE market-making PnL while protecting
# ~$8.7k of options gain.  Dialing the hedge down lets VE trade more
# freely; the options carry residual delta exposure scaled by
# (1 - HEDGE_RATIO).  At 0.5 we hedge half, leaving ~75 contracts of
# net option-delta exposure (manageable for typical intraday VE moves).
HEDGE_RATIO = 0.5


# ── Delta-one calls (VEV_4000) ────────────────────────────
# Tighter than VE because intrinsic is a much firmer anchor than VE's
# blended ANCHOR/EMA fair.  Cap is intentionally modest because each
# unit of D1 position pushes ~1.0 of synthetic delta into the VE hedge.
D1_EDGE = 2          # take when |market - intrinsic| > edge
D1_SIZE = 20
D1_POST_EDGE = 6     # post 3 ticks from intrinsic (was 1; only counterparty Mark 38 trades at ±2-3, so ±3 should still fill while tripling captured edge)
D1_POST_SIZE = 20
D1_CAP = 80          # per-strike position cap (vs 300 hard limit)
D1_SKEW_DIV = 30


# ── Options (v10: short-only, per-strike sizing) ─────────
#
# Round-4 evidence:
#   - Mark 01 bought 4,636 contracts and sold 0 (pure long-vol, wrong)
#   - Mark 22 sold 4,954 contracts and bought 18 (pure short-vol, right)
#   - Every long position we took in last run lost or barely earned
#   - Every short position made money; 5200 & 5300 pinned at -60 cap
#
# v10 changes:
#   - Options are SHORT-ONLY.  No bids posted, no take on offer.
#   - Per-strike short caps (was uniform 60).  Bigger caps where Mark 22
#     dumps the most volume.
#   - Larger sell-side sizes (was 10/10) since long path is gone.
#   - Far-OTM (6000, 6500) override fair to ~$0.5 so we post asks at $1
#     and harvest the rare uninformed lift (book is pinned 0/1).
#
# Round-4 evidence:
#   - Mark 01 bought 4,636 contracts and sold 0 (pure long-vol, wrong)
#   - Mark 22 sold 4,954 contracts and bought 18 (pure short-vol, right)
#   - Every long position we took in last run lost or barely earned
#   - Every short position made money; 5200 & 5300 pinned at -60 cap
#
# v10 changes:
#   - Options are SHORT-ONLY.  No bids posted, no take on offer.
#   - Per-strike short caps (was uniform 60).  Bigger caps where Mark 22
#     dumps the most volume.
#   - Larger sell-side sizes (was 10/10) since long path is gone.
#   - Far-OTM (6000, 6500) override fair to ~$0.5 so we post asks at $1
#     and harvest the rare uninformed lift (book is pinned 0/1).

OPTION_TAKE_SIZE = 20        # size when crossing (sell side only now)
OPTION_POST_SIZE = 15        # size of resting passive sells
OPTION_LONG_CAP = 0          # no longs allowed (short-only)

# Per-strike short caps — scaled to observed Mark-22 daily volume.
OPTION_SHORT_CAP: Dict[int, int] = {
    5200: 300,   # was hitting 60 cap, $1626 PnL → expect ~linear scale
    5300: 300,   # was hitting 60 cap, $1074 PnL → same
    5400: 300,
    5500: 300,
    6000: 100,   # rare fills, low premium
    6500: 100,
}

OPTION_BUY_EDGE  = 999.0     # effectively disabled (short-only)
OPTION_SELL_EDGE = 0.3       # cross to sell on small edge
OPTION_POST_BUY_EDGE  = 999.0  # disabled
OPTION_POST_SELL_EDGE = 0.3  # resting ask close to fair

OPTION_SKEW_DIV = 30         # inventory skew divisor on resting quotes

# Far-OTM fair override.  Smile fit gives ~$3 fair on 6000 because IV
# at the wing is high, but the actual book is permanently pinned 0/1.
# Override fair to $0.5 for these strikes so we post asks at $1 and
# join the existing offer rather than hovering uselessly higher.
OPTION_FAR_OTM_STRIKES = {6000, 6500}
OPTION_FAR_OTM_FAIR = 0.5

# Smile fit: force symmetric quadratic iv = a*m^2 + c
# (data shows linear coef is ~0.0005, indistinguishable from zero, so
# fitting it just adds noise from sparse wing strikes)
SMILE_SYMMETRIC = True

# Fallback smile when fit fails — calibrated from 3 days of data.
FALLBACK_SMILE_A = 1.90      # curvature
FALLBACK_SMILE_C = 0.017     # ATM IV
FALLBACK_SIGMA = 0.017       # used for delta-hedge calc

DAY_LENGTH = 1_000_000.0
STARTING_TTE = 5.0
OPTION_ENTRY_CUTOFF = 850_000


# ── Black-Scholes helpers ─────────────────────────────────

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_call_price(spot: float, strike: float, tte: float, sigma: float) -> float:
    if spot <= 0:
        return 0.0
    intrinsic = max(0.0, spot - strike)
    if tte <= 0 or sigma <= 0:
        return intrinsic
    try:
        sig_t = sigma * sqrt(tte)
        d1 = (log(spot / strike) + 0.5 * sigma * sigma * tte) / sig_t
        d2 = d1 - sig_t
        return spot * norm_cdf(d1) - strike * norm_cdf(d2)
    except Exception:
        return intrinsic


def bs_delta(spot: float, strike: float, tte: float, sigma: float) -> float:
    """Call delta = N(d1).  Used for hedging."""
    if spot <= 0:
        return 0.0
    if tte <= 0 or sigma <= 0:
        return 1.0 if spot > strike else 0.0
    try:
        sig_t = sigma * sqrt(tte)
        d1 = (log(spot / strike) + 0.5 * sigma * sigma * tte) / sig_t
        return norm_cdf(d1)
    except Exception:
        return 1.0 if spot > strike else 0.0


def implied_vol(target: float, spot: float, strike: float, tte: float) -> Optional[float]:
    """Bisection on sigma. None if target outside arbitrage bounds."""
    intrinsic = max(0.0, spot - strike)
    if target <= intrinsic + 1e-6 or target >= spot:
        return None
    lo, hi = 1e-4, 2.0
    if bs_call_price(spot, strike, tte, hi) < target:
        return None
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if bs_call_price(spot, strike, tte, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fit_smile(xs: List[float], ys: List[float],
              symmetric: bool = False) -> Optional[Tuple[float, float, float]]:
    """
    Returns (a, b, c) for iv = a*m^2 + b*m + c.

    If symmetric=True, forces b=0 and fits iv = a*m^2 + c.  This is more
    robust when the data shows the smile is symmetric (b ~ 0) and wing
    points are noisy.  Needs >= 2 points (>= 3 for asymmetric).
    """
    n = len(xs)

    if symmetric:
        if n < 2:
            return None
        # Solve for [a, c] minimizing sum (y - a*x^2 - c)^2.
        # Normal eqns: a*S(x^4) + c*S(x^2) = S(x^2 y)
        #              a*S(x^2) + c*n      = S(y)
        s2 = sum(x * x for x in xs)
        s4 = sum(x ** 4 for x in xs)
        t0 = sum(ys)
        t2 = sum(x * x * y for x, y in zip(xs, ys))
        det = s4 * float(n) - s2 * s2
        if abs(det) < 1e-12:
            return None
        a = (t2 * float(n) - t0 * s2) / det
        c = (s4 * t0 - s2 * t2) / det
        return a, 0.0, c

    if n < 3:
        return None
    s1 = sum(xs)
    s2 = sum(x * x for x in xs)
    s3 = sum(x ** 3 for x in xs)
    s4 = sum(x ** 4 for x in xs)
    t0 = sum(ys)
    t1 = sum(x * y for x, y in zip(xs, ys))
    t2 = sum(x * x * y for x, y in zip(xs, ys))
    M = [[s4, s3, s2], [s3, s2, s1], [s2, s1, float(n)]]
    rhs = [t2, t1, t0]

    def det3(A: List[List[float]]) -> float:
        return (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
                - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
                + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))

    D = det3(M)
    if abs(D) < 1e-12:
        return None
    out = []
    for j in range(3):
        Mj = [row[:] for row in M]
        for i in range(3):
            Mj[i][j] = rhs[i]
        out.append(det3(Mj) / D)
    return out[0], out[1], out[2]


class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        trader_data = self.load_trader_data(state.traderData)

        # ── HYDROGEL_PACK ─────────────────────────────────
        if TRADE_HYDROGEL:
            od = state.order_depths.get("HYDROGEL_PACK")
            if od is not None:
                pos = state.position.get("HYDROGEL_PACK", 0)
                # Compute HG Mark flow safely — never break the trader.
                try:
                    hg_flow = compute_hg_flow(state, trader_data)
                except Exception:
                    hg_flow = 0.0
                result["HYDROGEL_PACK"] = self.trade_hydrogel(
                    od, pos, trader_data, mark_flow=hg_flow,
                )
        elif "HYDROGEL_PACK" in state.order_depths:
            result["HYDROGEL_PACK"] = []

        # ── VELVETFRUIT_EXTRACT (with delta hedge from options) ──
        if TRADE_UNDERLYING:
            od = state.order_depths.get(UNDERLYING)
            if od is not None:
                pos = state.position.get(UNDERLYING, 0)

                # Compute net delta exposure from current option positions.
                hedge_offset = 0
                spot_for_delta = self.get_mid_price(od)
                if spot_for_delta is not None and spot_for_delta > 0:
                    tte_h = max(0.01, STARTING_TTE - state.timestamp / DAY_LENGTH)
                    net_d = 0.0
                    for strike in ACTIVE_STRIKES:
                        opos = state.position.get(f"VEV_{strike}", 0)
                        if opos != 0:
                            net_d += opos * bs_delta(
                                spot_for_delta, strike, tte_h, FALLBACK_SIGMA
                            )
                    hedge_offset = int(round(net_d * HEDGE_RATIO))

                # Compute Mark flow safely — never let it break the trader.
                try:
                    ve_flow = compute_ve_flow(state, trader_data)
                except Exception:
                    ve_flow = 0.0

                result[UNDERLYING] = self.trade_delta_one(
                    UNDERLYING, od, pos, POSITION_LIMITS[UNDERLYING], trader_data,
                    hedge_offset=hedge_offset,
                    mark_flow=ve_flow,
                )
        elif UNDERLYING in state.order_depths:
            result[UNDERLYING] = []

        # ── Delta-one calls (VEV_4000) ────────────────────
        # Treated as a synthetic underlying: fair = max(0, spot - K).
        # Positions feed the VE hedge_offset above on the next tick.
        if TRADE_OPTIONS:
            spot_for_d1 = self.get_mid_price(state.order_depths.get(UNDERLYING))
            if spot_for_d1 is not None and spot_for_d1 > 0:
                for strike in DELTA_ONE_STRIKES:
                    product = f"VEV_{strike}"
                    od = state.order_depths.get(product)
                    if od is None:
                        continue
                    pos = state.position.get(product, 0)
                    result[product] = self.trade_delta_one_call(
                        product, strike, od, pos, spot_for_d1,
                    )

        # ── Options with smile fit ────────────────────────
        if TRADE_OPTIONS:
            spot = self.get_mid_price(state.order_depths.get(UNDERLYING))
            if spot is not None and spot > 0:
                tte = max(0.01, STARTING_TTE - state.timestamp / DAY_LENGTH)
                sqrt_t = sqrt(tte)

                # Pass 1: invert IV from each strike's mid.
                xs: List[float] = []
                ys: List[float] = []
                for strike in ACTIVE_STRIKES:
                    od = state.order_depths.get(f"VEV_{strike}")
                    mid = self.get_mid_price(od)
                    if mid is None:
                        continue
                    iv = implied_vol(mid, spot, strike, tte)
                    if iv is None:
                        continue
                    xs.append(log(strike / spot) / sqrt_t)  # moneyness
                    ys.append(iv)

                fit = fit_smile(xs, ys, symmetric=SMILE_SYMMETRIC)

                # Pass 2: trade each strike against fitted IV.
                for strike in ACTIVE_STRIKES:
                    product = f"VEV_{strike}"
                    od = state.order_depths.get(product)
                    if od is None:
                        result[product] = []
                        continue

                    # v10: only quote strikes in TRADE_STRIKES.  Other
                    # strikes still contribute to the smile fit above
                    # but we don't post or take on them here.  Delta-one
                    # strikes are handled by trade_delta_one_call above.
                    if strike not in TRADE_STRIKES or strike in DELTA_ONE_STRIKES:
                        if product not in result:
                            result[product] = []
                        continue

                    m = log(strike / spot) / sqrt_t
                    if fit is not None:
                        a, b, c = fit
                        sigma_use = a * m * m + b * m + c
                    else:
                        # Use calibrated fallback smile, not flat sigma.
                        sigma_use = FALLBACK_SMILE_A * m * m + FALLBACK_SMILE_C
                    sigma_use = max(0.001, min(0.5, sigma_use))

                    pos = state.position.get(product, 0)
                    # v10: short-only on every traded strike.
                    result[product] = self.trade_option(
                        product, strike, od, pos, spot, tte, sigma_use,
                        state.timestamp,
                    )

        # Empty lists for inactive vouchers.
        for strike in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]:
            product = f"VEV_{strike}"
            if product in state.order_depths and product not in result:
                result[product] = []

        return result, 0, json.dumps(trader_data, separators=(",", ":"))

    # ─────────────────────────────────────────────────────
    def trade_hydrogel(self, od: OrderDepth, position: int, trader_data: Dict,
                       mark_flow: float = 0.0) -> List[Order]:
        """HG market-making with optional Mark-flow defence.

        mark_flow > 0: informed Marks have been buying.  Widen our
        SELL threshold (refuse to sell to them) but do not narrow our
        BUY threshold.

        mark_flow < 0: informed Marks have been selling.  Widen our
        BUY threshold (refuse to buy from them) but do not narrow our
        SELL threshold.
        """
        orders: List[Order] = []
        best_bid = self.best_bid(od)
        best_ask = self.best_ask(od)
        if best_bid is None or best_ask is None:
            return orders

        buy_room = POSITION_LIMITS["HYDROGEL_PACK"] - position
        sell_room = POSITION_LIMITS["HYDROGEL_PACK"] + position

        mid = (best_bid + best_ask) / 2.0

        # Fast EMA for momentum
        old_ema = trader_data.get(HG_MOM_KEY, mid)
        ema = (1.0 - HG_MOM_ALPHA) * old_ema + HG_MOM_ALPHA * mid
        trader_data[HG_MOM_KEY] = ema

        # Slow EMA for drifting anchor (v6).  Lets fair track sustained moves.
        old_slow = trader_data.get(HG_SLOW_KEY, mid)
        slow_ema = (1.0 - HG_SLOW_EMA_ALPHA) * old_slow + HG_SLOW_EMA_ALPHA * mid
        trader_data[HG_SLOW_KEY] = slow_ema

        raw_momentum = mid - ema
        mom_shift = max(-HG_MOM_CAP, min(HG_MOM_CAP, HG_MOM_K * raw_momentum))

        # v14c: ONLINE REGIME DETECTION.
        # Track the realized HG mid range over the last N ticks.
        # If realized range is wide (trending market), use slow-EMA
        # blend.  If narrow (flat market), use simple anchor.
        #
        # Implementation: keep rolling min/max with exponential decay
        # so old extremes fade.  When max-min > REGIME_THRESHOLD, we're
        # in a trending regime; below that, flat.
        REGIME_THRESHOLD = 25.0
        REGIME_DECAY = 0.998  # min/max decay 0.2% per tick toward mid

        regime_max_key = "HG_regime_max"
        regime_min_key = "HG_regime_min"
        old_max = trader_data.get(regime_max_key, mid)
        old_min = trader_data.get(regime_min_key, mid)
        # Decay toward mid, then update with new observation
        decayed_max = mid + (old_max - mid) * REGIME_DECAY
        decayed_min = mid + (old_min - mid) * REGIME_DECAY
        new_max = max(decayed_max, mid)
        new_min = min(decayed_min, mid)
        trader_data[regime_max_key] = new_max
        trader_data[regime_min_key] = new_min

        regime_range = new_max - new_min
        if regime_range > REGIME_THRESHOLD:
            # Trending regime → use slow-EMA blend
            anchor = HG_ANCHOR_WEIGHT * HG_FAIR + (1.0 - HG_ANCHOR_WEIGHT) * slow_ema
        else:
            # Flat regime → use fixed anchor
            anchor = HG_FAIR
        fair = anchor + mom_shift

        buy_take_edge = HG_TAKE_EDGE - max(0.0, mom_shift) * HG_MOM_TAKE_BOOST
        sell_take_edge = HG_TAKE_EDGE + min(0.0, mom_shift) * HG_MOM_TAKE_BOOST
        buy_take_edge = max(12.0, buy_take_edge)
        sell_take_edge = max(12.0, sell_take_edge)

        # ★ One-sided defensive bias from Mark flow.  Only WIDEN the
        # threshold on the side that informed Marks are pushing.
        if mark_flow > 0:
            sell_take_edge += HG_DEFENSIVE_BIAS * mark_flow
        elif mark_flow < 0:
            buy_take_edge += HG_DEFENSIVE_BIAS * (-mark_flow)

        if best_ask < fair - buy_take_edge and buy_room > 0:
            qty = min(HG_TAKE_SIZE, buy_room, abs(od.sell_orders[best_ask]))
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", best_ask, qty))
                position += qty
                buy_room -= qty

        if best_bid > fair + sell_take_edge and sell_room > 0:
            qty = min(HG_TAKE_SIZE, sell_room, od.buy_orders[best_bid])
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", best_bid, -qty))
                position -= qty
                sell_room -= qty

        skew = self.signed_bucket(position, HG_SKEW_DIV)
        bid_px = floor(fair - HG_POST_EDGE - skew)
        ask_px = ceil(fair + HG_POST_EDGE - skew)

        if buy_room > 0 and bid_px < best_ask:
            qty = min(HG_POST_SIZE, buy_room)
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", bid_px, qty))

        if sell_room > 0 and ask_px > best_bid:
            qty = min(HG_POST_SIZE, sell_room)
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", ask_px, -qty))

        return orders

    # ─────────────────────────────────────────────────────
    def trade_delta_one(
        self, product: str, od: OrderDepth, position: int, limit: int,
        trader_data: Dict, hedge_offset: int = 0, mark_flow: float = 0.0,
    ) -> List[Order]:
        """
        hedge_offset: net delta exposure from option positions.  Treated as
        synthetic inventory when computing skew, so the underlying market
        maker leans against the total delta (underlying + options).

        mark_flow: signed informed-Mark flow on this product in [-1, +1].
        Used ONLY to bias take edges (don't get picked off by smart Marks).
        Fair value is unchanged.
        """
        orders: List[Order] = []
        best_bid = self.best_bid(od)
        best_ask = self.best_ask(od)
        if best_bid is None or best_ask is None:
            return orders

        mid = (best_bid + best_ask) / 2.0
        ema_key = f"{product}_ema"
        old_ema = trader_data.get(ema_key, mid)
        ema = (1.0 - VE_EMA_ALPHA) * old_ema + VE_EMA_ALPHA * mid
        trader_data[ema_key] = ema

        blended = VE_ANCHOR_WEIGHT * VE_ANCHOR + (1.0 - VE_ANCHOR_WEIGHT) * ema

        bid_vol = float(od.buy_orders[best_bid])
        ask_vol = float(abs(od.sell_orders[best_ask]))
        total = bid_vol + ask_vol

        if total > 0:
            imbalance = (bid_vol - ask_vol) / total
            shift = max(-VE_IMBALANCE_CAP, min(VE_IMBALANCE_CAP, VE_IMBALANCE_K * imbalance))
        else:
            shift = 0.0

        fair = blended + shift
        buy_room = limit - position
        sell_room = limit + position

        # ★ One-sided defensive bias from Mark flow on VE.  Only WIDEN
        # thresholds; never narrow.  This avoids the v5b failure mode
        # of chasing signal at bad prices.
        buy_edge = VE_EDGE
        sell_edge = VE_EDGE
        if mark_flow > 0:
            sell_edge += VE_DEFENSIVE_BIAS * mark_flow
        elif mark_flow < 0:
            buy_edge += VE_DEFENSIVE_BIAS * (-mark_flow)

        if best_ask < fair - buy_edge and buy_room > 0:
            qty = min(VE_SIZE, buy_room, abs(od.sell_orders[best_ask]))
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
                position += qty
                buy_room -= qty

        if best_bid > fair + sell_edge and sell_room > 0:
            qty = min(VE_SIZE, sell_room, od.buy_orders[best_bid])
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
                position -= qty
                sell_room -= qty

        # Skew uses synthetic position = real + option delta.
        synthetic_pos = position + hedge_offset
        skew = synthetic_pos // VE_SKEW_DIV
        bid_px = int(fair - VE_POST_EDGE - skew)
        ask_px = int(fair + VE_POST_EDGE - skew)

        if buy_room > 0 and bid_px < best_ask:
            qty = min(VE_POST_SIZE, buy_room)
            if qty > 0:
                orders.append(Order(product, bid_px, qty))

        if sell_room > 0 and ask_px > best_bid:
            qty = min(VE_POST_SIZE, sell_room)
            if qty > 0:
                orders.append(Order(product, ask_px, -qty))

        return orders

    # ─────────────────────────────────────────────────────
    def trade_delta_one_call(
        self, product: str, strike: int, od: OrderDepth, position: int,
        spot: float,
    ) -> List[Order]:
        """Market-make a deep-ITM call as a synthetic underlying.

        Fair value = max(0, spot - strike).  Time value is empirically
        zero (avg $0.008 over 3 days), so BS pricing is unnecessary.
        Positions push synthetic delta into the VE hedge_offset on the
        next tick, so risk is auto-hedged in the underlying.
        """
        orders: List[Order] = []
        best_bid = self.best_bid(od)
        best_ask = self.best_ask(od)
        if best_bid is None or best_ask is None:
            return orders

        fair = max(0.0, spot - strike)

        buy_room = min(POSITION_LIMITS[product] - position, D1_CAP - position)
        sell_room = min(POSITION_LIMITS[product] + position, D1_CAP + position)
        buy_room = max(0, buy_room)
        sell_room = max(0, sell_room)

        # ── Aggressive take when market diverges from intrinsic ──
        if best_ask < fair - D1_EDGE and buy_room > 0:
            qty = min(D1_SIZE, buy_room, abs(od.sell_orders[best_ask]))
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
                position += qty
                buy_room -= qty

        if best_bid > fair + D1_EDGE and sell_room > 0:
            qty = min(D1_SIZE, sell_room, od.buy_orders[best_bid])
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
                position -= qty
                sell_room -= qty

        # ── Passive quotes around intrinsic, with inventory skew ──
        skew = self.signed_bucket(position, D1_SKEW_DIV)
        bid_px = int(fair - D1_POST_EDGE - skew)
        ask_px = int(fair + D1_POST_EDGE - skew)

        if buy_room > 0 and bid_px < best_ask and bid_px > 0:
            qty = min(D1_POST_SIZE, buy_room)
            if qty > 0:
                orders.append(Order(product, bid_px, qty))

        if sell_room > 0 and ask_px > best_bid:
            qty = min(D1_POST_SIZE, sell_room)
            if qty > 0:
                orders.append(Order(product, ask_px, -qty))

        return orders

    # ─────────────────────────────────────────────────────
    def trade_option(
        self, product: str, strike: int, od: OrderDepth, position: int,
        spot: float, tte: float, sigma: float, timestamp: int,
    ) -> List[Order]:
        """v10: SHORT-ONLY market making against smile-fitted fair.

        - No bids posted, no take on offer.  Long-side bled in every
          run; short-side is the only consistent winner.
        - Per-strike short caps (OPTION_SHORT_CAP).
        - Far-OTM strikes (6000, 6500) override fair to $0.5 to post
          asks at $1, joining the permanent offer in the book.
        - Existing longs (from prior runs / boot state) can still be
          flushed in EXIT MODE.
        """
        orders: List[Order] = []
        best_bid = self.best_bid(od)
        best_ask = self.best_ask(od)
        if best_bid is None or best_ask is None:
            return orders

        # Compute fair (smile-fit), then override for far OTM.
        fair = bs_call_price(spot, strike, tte, sigma)
        if strike in OPTION_FAR_OTM_STRIKES:
            fair = OPTION_FAR_OTM_FAIR

        short_cap = OPTION_SHORT_CAP.get(strike, 60)
        # buy_room: only used to flush an accidental long in EXIT MODE.
        buy_room = min(
            POSITION_LIMITS[product] - position,
            OPTION_LONG_CAP - position,   # = -position when LONG_CAP=0
        )
        buy_room = max(0, buy_room)
        sell_room = min(
            POSITION_LIMITS[product] + position,
            short_cap + position,
        )
        sell_room = max(0, sell_room)

        # ── EXIT MODE near end of horizon ─────────────────
        if timestamp >= OPTION_ENTRY_CUTOFF:
            if position > 0:
                qty = min(position, OPTION_TAKE_SIZE, od.buy_orders[best_bid])
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
            elif position < 0:
                qty = min(-position, OPTION_TAKE_SIZE, abs(od.sell_orders[best_ask]))
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
            return orders

        # ── 1. AGGRESSIVE TAKE — sell only ────────────────
        # Sell triggers on small edge (shorts collect theta + vol drift).
        if best_bid > fair + OPTION_SELL_EDGE and sell_room > 0:
            qty = min(OPTION_TAKE_SIZE, sell_room, od.buy_orders[best_bid])
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
                position -= qty
                sell_room -= qty

        # ── 2. PASSIVE QUOTES — sell side only ────────────
        skew = position / OPTION_SKEW_DIV
        adj_fair = fair - skew

        # Ask: closer to fair, joining or improving the offer.
        if sell_room > 0:
            target_ask = ceil(adj_fair + OPTION_POST_SELL_EDGE)
            improve = best_ask - 1
            if improve > adj_fair + OPTION_POST_SELL_EDGE and improve > best_bid:
                target_ask = improve
            # For far-OTM, also allow joining best_ask directly.
            if strike in OPTION_FAR_OTM_STRIKES:
                target_ask = max(1, min(target_ask, best_ask))
            if target_ask > best_bid:
                qty = min(OPTION_POST_SIZE, sell_room)
                if qty > 0:
                    orders.append(Order(product, target_ask, -qty))

        return orders

    # ─────────────────────────────────────────────────────
    def load_trader_data(self, raw: str) -> Dict:
        if not raw:
            return {}
        try:
            d = json.loads(raw)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def best_bid(od: Optional[OrderDepth]) -> Optional[int]:
        if od is None or not od.buy_orders:
            return None
        return max(od.buy_orders)

    @staticmethod
    def best_ask(od: Optional[OrderDepth]) -> Optional[int]:
        if od is None or not od.sell_orders:
            return None
        return min(od.sell_orders)

    @staticmethod
    def get_mid_price(od: Optional[OrderDepth]) -> Optional[float]:
        if od is None:
            return None
        bb = max(od.buy_orders) if od.buy_orders else None
        ba = min(od.sell_orders) if od.sell_orders else None
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        if bb is not None:
            return float(bb)
        if ba is not None:
            return float(ba)
        return None

    @staticmethod
    def signed_bucket(position: int, divisor: int) -> int:
        if divisor <= 0:
            return 0
        return int(position / divisor)