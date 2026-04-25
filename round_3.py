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
TRADE_OPTIONS = False
TRADE_HYDROGEL = True

UNDERLYING = "VELVETFRUIT_EXTRACT"

ACTIVE_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]


# ── HYDROGEL_PACK (unchanged) ─────────────────────────────

HG_FAIR = 9991
HG_TAKE_EDGE = 26
HG_TAKE_SIZE = 20
HG_POST_EDGE = 8
HG_POST_SIZE = 20
HG_SKEW_DIV = 40

# ── HYDROGEL momentum overlay ─────────────────────────────
# Keeps HG_FAIR as the anchor, but nudges fair when recent mids trend.
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

OPTION_TAKE_SIZE = 10        # size when crossing
OPTION_POST_SIZE = 10        # size of resting passive quotes
OPTION_MAX_POSITION = 60     # per-strike soft cap

OPTION_TAKE_EDGE = 0.5       # cross when |market_px - smile_px| > this
OPTION_POST_EDGE = 0.5       # min distance from fair for resting quotes
OPTION_SKEW_DIV = 30         # inventory skew divisor on resting quotes

FALLBACK_SIGMA = 0.017       # used when smile fit fails (data: 0.016-0.019)
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
    """LS quadratic fit y = a*x^2 + b*x + c.  Needs >= 3 points."""
    n = len(xs)
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

                fit = fit_smile(xs, ys)

                # Pass 2: trade each strike against fitted IV.
                for strike in ACTIVE_STRIKES:
                    product = f"VEV_{strike}"
                    od = state.order_depths.get(product)
                    if od is None:
                        result[product] = []
                        continue

                    if fit is not None:
                        m = log(strike / spot) / sqrt_t
                        a, b, c = fit
                        sigma_use = max(0.001, min(0.5, a * m * m + b * m + c))
                    else:
                        sigma_use = FALLBACK_SIGMA

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

        # ── Momentum overlay ──────────────────────────────
        # mid > EMA means short-term upward momentum.
        # mid < EMA means short-term downward momentum.
        mid = (best_bid + best_ask) / 2.0
        old_ema = trader_data.get(HG_MOM_KEY, mid)
        ema = (1.0 - HG_MOM_ALPHA) * old_ema + HG_MOM_ALPHA * mid
        trader_data[HG_MOM_KEY] = ema

        raw_momentum = mid - ema
        mom_shift = max(-HG_MOM_CAP, min(HG_MOM_CAP, HG_MOM_K * raw_momentum))
        fair = HG_FAIR + mom_shift

        # With upward momentum, buy slightly more aggressively.
        # With downward momentum, sell slightly more aggressively.
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
        """Take + post against a smile-fitted fair value."""
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

        # ── 1. AGGRESSIVE TAKE on mispricings ─────────────
        if best_ask < fair - OPTION_TAKE_EDGE and buy_room > 0:
            qty = min(OPTION_TAKE_SIZE, buy_room, abs(od.sell_orders[best_ask]))
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
                position += qty
                buy_room -= qty

        if best_bid > fair + OPTION_TAKE_EDGE and sell_room > 0:
            qty = min(OPTION_TAKE_SIZE, sell_room, od.buy_orders[best_bid])
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
                position -= qty
                sell_room -= qty

        # ── 2. PASSIVE QUOTES — join or improve the inside ─
        skew = position / OPTION_SKEW_DIV  # float skew, applied to fair
        adj_fair = fair - skew

        # Bid: target one tick inside the spread, but never above fair-edge
        if buy_room > 0:
            target_bid = floor(adj_fair - OPTION_POST_EDGE)
            # If we can improve (best_bid + 1 still leaves edge), do so
            improve = best_bid + 1
            if improve < adj_fair - OPTION_POST_EDGE and improve < best_ask:
                target_bid = improve
            if target_bid >= 0 and target_bid < best_ask:
                qty = min(OPTION_POST_SIZE, buy_room)
                if qty > 0:
                    orders.append(Order(product, target_bid, qty))

        # Ask: symmetric
        if sell_room > 0:
            target_ask = ceil(adj_fair + OPTION_POST_EDGE)
            improve = best_ask - 1
            if improve > adj_fair + OPTION_POST_EDGE and improve > best_bid:
                target_ask = improve
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