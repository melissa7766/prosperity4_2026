from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional
from math import ceil, erf, floor, log, sqrt
import json


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

# Restrict to strikes near spot — wings (4000/4500/6000/6500) have illiquid
# books and corrupt the smile fit with noise.
# Use liquid near-spot strikes for fitting, but only trade the safer core.
# 5400/5500 are low-price/noisy; passive fills there caused bleed.
FIT_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]
TRADED_OPTION_STRIKES = [5000, 5100, 5200, 5300]


# ── HYDROGEL_PACK (unchanged) ─────────────────────────────

HG_FAIR = 9991
HG_TAKE_EDGE = 26
HG_TAKE_SIZE = 20
HG_POST_EDGE = 8
HG_POST_SIZE = 20
HG_SKEW_DIV = 25

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

VE_EDGE = 7
VE_SIZE = 20
VE_POST_EDGE = 1
VE_POST_SIZE = 20


# ── Options ───────────────────────────────────────────────

OPTION_TAKE_SIZE = 8
OPTION_POST_SIZE = 5
OPTION_MAX_POSITION = 35

# Asymmetric edges — bias toward fading vol (sell options).
# Need real edge. The old values were too tight and got picked off.
OPTION_BUY_EDGE  = 2.0
OPTION_SELL_EDGE = 1.5
OPTION_POST_BUY_EDGE  = 2.0
OPTION_POST_SELL_EDGE = 2.0

OPTION_SKEW_DIV = 30

# Close-out: when more than half the cap, take favorable prices to lock gains.
OPTION_CLOSE_THRESHOLD = OPTION_MAX_POSITION // 2
OPTION_CLOSE_EDGE = 0.0      # only close when market is at fair OR BETTER
                             # (was 0.5 — paid spread to close, ate the edge)

# Skip strikes where spread is too wide vs price.  On 5500 (~18% spread)
# round-trips eat the edge.  Quote passively only on those — never take.
OPTION_MAX_SPREAD_PCT = 0.08  # 8% of mid
OPTION_MIN_PRICE = 20         # also skip taking if option price < this

# Fallback smile when fit fails — calibrated from 3 days of historical data.
# IV ≈ A * m^2 + C  where m = log(K/S)/sqrt(T)
# More realistic fallback from the 3 historical days.
# Old C=0.017 overvalued options and made the bot buy noisy wings.
FALLBACK_SMILE_A = 0.55
FALLBACK_SMILE_C = 0.0137

DAY_LENGTH = 1_000_000.0
STARTING_TTE = 5.0
OPTION_ENTRY_CUTOFF = 800_000


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


def fit_smile(xs: List[float], ys: List[float]) -> Optional[Tuple[float, float, float]]:
    """
    Fit iv = a*m^2 + b*m + c.
    Uses a tiny 3x3 least-squares solve and rejects unstable fits.
    The old symmetric fit often produced negative curvature, which made
    5400 look cheap and 5200/5300 look too expensive.
    """
    n = len(xs)
    if n < 4:
        return None

    s0 = float(n)
    s1 = sum(xs)
    s2 = sum(x * x for x in xs)
    s3 = sum(x ** 3 for x in xs)
    s4 = sum(x ** 4 for x in xs)
    y0 = sum(ys)
    y1 = sum(x * y for x, y in zip(xs, ys))
    y2 = sum(x * x * y for x, y in zip(xs, ys))

    # Solve normal equations for [a, b, c].
    A = [[s4, s3, s2], [s3, s2, s1], [s2, s1, s0]]
    bvec = [y2, y1, y0]

    try:
        # Manual Gaussian elimination to avoid importing numpy.
        for i in range(3):
            pivot = i
            for r in range(i + 1, 3):
                if abs(A[r][i]) > abs(A[pivot][i]):
                    pivot = r
            if abs(A[pivot][i]) < 1e-12:
                return None
            if pivot != i:
                A[i], A[pivot] = A[pivot], A[i]
                bvec[i], bvec[pivot] = bvec[pivot], bvec[i]
            div = A[i][i]
            for j in range(i, 3):
                A[i][j] /= div
            bvec[i] /= div
            for r in range(3):
                if r == i:
                    continue
                mult = A[r][i]
                for j in range(i, 3):
                    A[r][j] -= mult * A[i][j]
                bvec[r] -= mult * bvec[i]
        a, b, c = bvec[0], bvec[1], bvec[2]
    except Exception:
        return None

    # Reject noisy/overfit smiles. Fallback is safer than a bad fit.
    if a < 0.0 or a > 2.5:
        return None
    if b < -0.06 or b > 0.06:
        return None
    if c < 0.0125 or c > 0.0150:
        return None
    return a, b, c


class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        trader_data = self.load_trader_data(state.traderData)

        # ── HYDROGEL_PACK ─────────────────────────────────
        if TRADE_HYDROGEL:
            od = state.order_depths.get("HYDROGEL_PACK")
            if od is not None:
                pos = state.position.get("HYDROGEL_PACK", 0)
                result["HYDROGEL_PACK"] = self.trade_hydrogel(od, pos, trader_data)
        elif "HYDROGEL_PACK" in state.order_depths:
            result["HYDROGEL_PACK"] = []

        # ── VELVETFRUIT_EXTRACT ───────────────────────────
        if TRADE_UNDERLYING:
            od = state.order_depths.get(UNDERLYING)
            if od is not None:
                pos = state.position.get(UNDERLYING, 0)
                result[UNDERLYING] = self.trade_delta_one(
                    UNDERLYING, od, pos, POSITION_LIMITS[UNDERLYING], trader_data
                )
        elif UNDERLYING in state.order_depths:
            result[UNDERLYING] = []

        # ── Options with smile fit ────────────────────────
        if TRADE_OPTIONS:
            spot = self.get_mid_price(state.order_depths.get(UNDERLYING))
            if spot is not None and spot > 0:
                tte = max(0.01, STARTING_TTE - state.timestamp / DAY_LENGTH)
                sqrt_t = sqrt(tte)

                # Pass 1: invert IV from each ACTIVE strike's mid.
                xs: List[float] = []
                ys: List[float] = []
                for strike in FIT_STRIKES:
                    od = state.order_depths.get(f"VEV_{strike}")
                    mid = self.get_mid_price(od)
                    if mid is None:
                        continue
                    iv = implied_vol(mid, spot, strike, tte)
                    if iv is None:
                        continue
                    xs.append(log(strike / spot) / sqrt_t)
                    ys.append(iv)

                fit = fit_smile(xs, ys)

                # Pass 2: trade each strike against fitted IV.
                for strike in TRADED_OPTION_STRIKES:
                    product = f"VEV_{strike}"
                    od = state.order_depths.get(product)
                    if od is None:
                        result[product] = []
                        continue

                    m = log(strike / spot) / sqrt_t
                    if fit is not None:
                        a, b, c = fit
                        sigma_use = a * m * m + b * m + c
                    else:
                        # Conservative fallback smile (not flat sigma).
                        sigma_use = FALLBACK_SMILE_A * m * m + FALLBACK_SMILE_C
                    sigma_use = max(0.001, min(0.5, sigma_use))

                    pos = state.position.get(product, 0)
                    result[product] = self.trade_option(
                        product, strike, od, pos, spot, tte, sigma_use, state.timestamp
                    )

        # Empty lists for inactive vouchers.
        for strike in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]:
            product = f"VEV_{strike}"
            if product in state.order_depths and product not in result:
                result[product] = []

        return result, 0, json.dumps(trader_data, separators=(",", ":"))

    # ─────────────────────────────────────────────────────
    def trade_hydrogel(self, od: OrderDepth, position: int, trader_data: Dict) -> List[Order]:
        orders: List[Order] = []
        best_bid = self.best_bid(od)
        best_ask = self.best_ask(od)
        if best_bid is None or best_ask is None:
            return orders

        buy_room = POSITION_LIMITS["HYDROGEL_PACK"] - position
        sell_room = POSITION_LIMITS["HYDROGEL_PACK"] + position

        mid = (best_bid + best_ask) / 2.0
        old_ema = trader_data.get(HG_MOM_KEY, mid)
        ema = (1.0 - HG_MOM_ALPHA) * old_ema + HG_MOM_ALPHA * mid
        trader_data[HG_MOM_KEY] = ema

        raw_momentum = mid - ema
        mom_shift = max(-HG_MOM_CAP, min(HG_MOM_CAP, HG_MOM_K * raw_momentum))
        fair = HG_FAIR + mom_shift

        buy_take_edge = HG_TAKE_EDGE - max(0.0, mom_shift) * HG_MOM_TAKE_BOOST
        sell_take_edge = HG_TAKE_EDGE + min(0.0, mom_shift) * HG_MOM_TAKE_BOOST
        buy_take_edge = max(12.0, buy_take_edge)
        sell_take_edge = max(12.0, sell_take_edge)

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

        skew = position // HG_SKEW_DIV
        bid_px = int(fair - HG_POST_EDGE - skew)
        ask_px = int(fair + HG_POST_EDGE - skew)

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
        trader_data: Dict,
    ) -> List[Order]:
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

        if best_ask < fair - VE_EDGE and buy_room > 0:
            qty = min(VE_SIZE, buy_room, abs(od.sell_orders[best_ask]))
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
                position += qty
                buy_room -= qty

        if best_bid > fair + VE_EDGE and sell_room > 0:
            qty = min(VE_SIZE, sell_room, od.buy_orders[best_bid])
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
                position -= qty
                sell_room -= qty

        skew = position // 34
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
    def trade_option(
        self, product: str, strike: int, od: OrderDepth, position: int,
        spot: float, tte: float, sigma: float, timestamp: int,
    ) -> List[Order]:
        """Take + post against a smile-fitted fair value, with close-out."""
        orders: List[Order] = []
        best_bid = self.best_bid(od)
        best_ask = self.best_ask(od)
        if best_bid is None or best_ask is None:
            return orders

        fair = bs_call_price(spot, strike, tte, sigma)

        buy_room = min(
            POSITION_LIMITS[product] - position,
            OPTION_MAX_POSITION - position,
        )
        sell_room = min(
            POSITION_LIMITS[product] + position,
            OPTION_MAX_POSITION + position,
        )

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

        # Spread guard — skip taking/closing on options where spread eats edge.
        # Still post passive quotes (they capture the spread when filled).
        spread = best_ask - best_bid
        mid = (best_ask + best_bid) / 2.0
        spread_too_wide = (mid > 0 and spread / mid > OPTION_MAX_SPREAD_PCT) \
                          or (mid < OPTION_MIN_PRICE)

        # ── 1. AGGRESSIVE TAKE on mispricings ─────────────
        if not spread_too_wide:
            if best_ask < fair - OPTION_BUY_EDGE and buy_room > 0:
                qty = min(OPTION_TAKE_SIZE, buy_room, abs(od.sell_orders[best_ask]))
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    position += qty
                    buy_room -= qty

            if best_bid > fair + OPTION_SELL_EDGE and sell_room > 0:
                qty = min(OPTION_TAKE_SIZE, sell_room, od.buy_orders[best_bid])
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    position -= qty
                    sell_room -= qty

        # ── 1b. CLOSE-OUT to lock gains ───────────────────
        # Without this, positions drift to cap and bleed theta while the
        # smile residual reverts.  When more than half the cap and market
        # is at or near fair, take the favorable side.
        # Skip close-out crossing the spread when spread is wide — would
        # cost more than the position is worth.  Let passive quotes do it.
        if not spread_too_wide:
            if position < -OPTION_CLOSE_THRESHOLD and best_ask <= fair + OPTION_CLOSE_EDGE:
                qty = min(-position, OPTION_TAKE_SIZE,
                          abs(od.sell_orders[best_ask]), buy_room)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    position += qty
                    buy_room -= qty

            if position > OPTION_CLOSE_THRESHOLD and best_bid >= fair - OPTION_CLOSE_EDGE:
                qty = min(position, OPTION_TAKE_SIZE,
                          od.buy_orders[best_bid], sell_room)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    position -= qty
                    sell_room -= qty

        # ── 2. PASSIVE QUOTES ─────────────────────────────
        # Do NOT quote both sides just because we have a fair value.
        # Only post when the current book is already mispriced by enough edge.
        # This reduces adverse-selection inventory in near-ATM vouchers.
        if spread_too_wide:
            return orders

        skew = position / OPTION_SKEW_DIV
        adj_fair = fair - skew

        # Join/improve the bid only if buying still has edge.
        if buy_room > 0 and best_bid < adj_fair - OPTION_POST_BUY_EDGE:
            target_bid = min(best_bid + 1, floor(adj_fair - OPTION_POST_BUY_EDGE))
            if target_bid >= 0 and target_bid < best_ask:
                qty = min(OPTION_POST_SIZE, buy_room)
                if qty > 0:
                    orders.append(Order(product, target_bid, qty))

        # Join/improve the ask only if selling still has edge.
        if sell_room > 0 and best_ask > adj_fair + OPTION_POST_SELL_EDGE:
            target_ask = max(best_ask - 1, ceil(adj_fair + OPTION_POST_SELL_EDGE))
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