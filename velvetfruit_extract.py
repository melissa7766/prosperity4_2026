from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional
from math import erf, log, sqrt
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
TRADE_OPTIONS    = True
UNDERLYING       = "VELVETFRUIT_EXTRACT"
ACTIVE_STRIKES   = []

# ── VFX fair price ────────────────────────────────────────
#
#   fair = ANCHOR_WEIGHT * ANCHOR
#        + (1 - ANCHOR_WEIGHT) * ema
#        + IMBALANCE_K * imbalance
#
#   ANCHOR       — stable mean from historical data (~5262).
#                  Adjust if VFX drifts materially day-to-day.
#   ANCHOR_WEIGHT — how hard fair is pulled back to anchor.
#                   1.0 = pure anchor; 0.0 = pure EMA.
#                   0.6 keeps a gentle EMA drift while anchoring.
#   EMA_ALPHA    — EMA speed. Only matters when ANCHOR_WEIGHT < 1.
#                  Higher = more responsive, more noise.
#   IMBALANCE_K  — how many ticks to shift fair per unit of
#                   order-book imbalance. Range [-1, 1].
#                   2.0 means ±2 tick shift at max imbalance.
#   IMBALANCE_CAP — clamps the imbalance shift (ticks).

VE_ANCHOR        = 5262.0
VE_ANCHOR_WEIGHT = 0.6
VE_EMA_ALPHA     = 0.08
VE_IMBALANCE_K   = 2.0
VE_IMBALANCE_CAP = 2.0

# ── VFX order placement ───────────────────────────────────
#
#   VE_EDGE  — min gap between fair and market price to TAKE.
#   VE_SIZE  — contracts per market order.
#   POST_EDGE — offset from fair to post passive limit orders.
#   POST_SIZE — contracts per passive order.
#
#   Passive orders earn the spread; market orders pay it.
#   With a good fair price, passive posting is the main edge.

VE_EDGE      = 7
VE_SIZE      = 20
VE_POST_EDGE = 1
VE_POST_SIZE = 20

# ── Options ───────────────────────────────────────────────
OPTION_EDGE         = 1.2
OPTION_SIZE         = 25
OPTION_MAX_POSITION = 150
SIGMA               = 0.0140   # live TTE=5 IV

DAY_LENGTH   = 1_000_000.0
STARTING_TTE = 5.0


# ── Black-Scholes ─────────────────────────────────────────
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def bs_call_price(spot: float, strike: float, tte_days: float, sigma: float) -> float:
    if spot <= 0:
        return 0.0
    intrinsic = max(0.0, spot - strike)
    if tte_days <= 0 or sigma <= 0:
        return intrinsic
    try:
        sigma_t = sigma * sqrt(tte_days)
        d1 = (log(spot / strike) + 0.5 * sigma * sigma * tte_days) / sigma_t
        d2 = d1 - sigma_t
        return spot * norm_cdf(d1) - strike * norm_cdf(d2)
    except Exception:
        return intrinsic


class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        trader_data = self.load_trader_data(state.traderData)

        if TRADE_UNDERLYING:
            od = state.order_depths.get(UNDERLYING)
            if od is not None:
                pos = state.position.get(UNDERLYING, 0)
                result[UNDERLYING] = self.trade_delta_one(
                    UNDERLYING, od, pos, POSITION_LIMITS[UNDERLYING], trader_data
                )

        if TRADE_OPTIONS:
            spot = self.get_mid_price(state.order_depths.get(UNDERLYING))
            if spot is not None:
                tte = max(0.01, STARTING_TTE - state.timestamp / DAY_LENGTH)
                for strike in ACTIVE_STRIKES:
                    product = f"VEV_{strike}"
                    od = state.order_depths.get(product)
                    if od is None:
                        result[product] = []
                        continue
                    pos = state.position.get(product, 0)
                    result[product] = self.trade_option(product, strike, od, pos, spot, tte)

        for strike in [4000, 4500, 5300, 5400, 5500, 6000, 6500]:
            product = f"VEV_{strike}"
            if product in state.order_depths and product not in result:
                result[product] = []

        if "HYDROGEL_PACK" in state.order_depths:
            result["HYDROGEL_PACK"] = []

        return result, 0, json.dumps(trader_data, separators=(",", ":"))

    # ─────────────────────────────────────────────────────
    def trade_delta_one(
        self,
        product: str,
        od: OrderDepth,
        position: int,
        limit: int,
        trader_data: Dict,
    ) -> List[Order]:
        orders: List[Order] = []

        best_bid = self.best_bid(od)
        best_ask = self.best_ask(od)
        if best_bid is None or best_ask is None:
            return orders

        mid = (best_bid + best_ask) / 2.0

        # ── Step 1: EMA (slow drift tracker) ─────────────
        ema_key = f"{product}_ema"
        ema = (1 - VE_EMA_ALPHA) * trader_data.get(ema_key, mid) + VE_EMA_ALPHA * mid
        trader_data[ema_key] = ema

        # ── Step 2: anchor blend ──────────────────────────
        #   Pulls fair toward the long-run mean.
        #   EMA handles short-term drift; anchor prevents runaway.
        blended = VE_ANCHOR_WEIGHT * VE_ANCHOR + (1 - VE_ANCHOR_WEIGHT) * ema

        # ── Step 3: microprice imbalance shift ────────────
        #   If buy-side depth >> sell-side depth, price likely
        #   to tick up → shift fair slightly higher, and vice versa.
        bid_vol = float(od.buy_orders[best_bid])
        ask_vol = float(abs(od.sell_orders[best_ask]))
        total   = bid_vol + ask_vol
        if total > 0:
            imbalance = (bid_vol - ask_vol) / total   # [-1, 1]
            shift = max(-VE_IMBALANCE_CAP, min(VE_IMBALANCE_CAP, VE_IMBALANCE_K * imbalance))
        else:
            shift = 0.0

        fair = blended + shift

        # ── Step 4: take orders (cross spread on big edges) ─
        buy_room  = limit - position
        sell_room = limit + position

        if best_ask < fair - VE_EDGE and buy_room > 0:
            qty = min(VE_SIZE, buy_room, abs(od.sell_orders[best_ask]))
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
                buy_room  -= qty
                position  += qty

        if best_bid > fair + VE_EDGE and sell_room > 0:
            qty = min(VE_SIZE, sell_room, od.buy_orders[best_bid])
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
                sell_room -= qty
                position  -= qty

        # ── Step 5: passive quotes (earn the spread) ──────
        #   Post just inside fair ± POST_EDGE, but never cross
        #   the existing best bid/ask (we'd get immediately filled
        #   and pay the spread instead of earning it).
        skew    = position // 34          # nudge quotes away from inventory
        bid_px  = int(fair - VE_POST_EDGE - skew)
        ask_px  = int(fair + VE_POST_EDGE - skew)

        if buy_room > 0 and bid_px < best_ask:
            orders.append(Order(product, bid_px, min(VE_POST_SIZE, buy_room)))

        if sell_room > 0 and ask_px > best_bid:
            orders.append(Order(product, ask_px, -min(VE_POST_SIZE, sell_room)))

        return orders

    # ─────────────────────────────────────────────────────
    def trade_option(
        self,
        product: str,
        strike: int,
        od: OrderDepth,
        position: int,
        spot: float,
        tte: float,
    ) -> List[Order]:
        orders: List[Order] = []
        best_bid = self.best_bid(od)
        best_ask = self.best_ask(od)
        if best_bid is None or best_ask is None:
            return orders

        fair = bs_call_price(spot, strike, tte, SIGMA)

        if fair - best_ask > OPTION_EDGE and position < OPTION_MAX_POSITION:
            buy_room = min(POSITION_LIMITS[product] - position, OPTION_MAX_POSITION - position)
            qty = min(OPTION_SIZE, buy_room, abs(od.sell_orders[best_ask]))
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
                position += qty

        if best_bid - fair > OPTION_EDGE and position > -OPTION_MAX_POSITION:
            sell_room = min(POSITION_LIMITS[product] + position, OPTION_MAX_POSITION + position)
            qty = min(OPTION_SIZE, sell_room, od.buy_orders[best_bid])
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))

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
        return float(bb) if bb is not None else (float(ba) if ba is not None else None)