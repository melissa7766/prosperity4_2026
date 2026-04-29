from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import math
import json


class Trader:


    POSITION_LIMIT = 10

    PRODUCTS = [
        "GALAXY_SOUNDS_DARK_MATTER", "GALAXY_SOUNDS_BLACK_HOLES",
        "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_WINDS",
        "GALAXY_SOUNDS_SOLAR_FLAMES",

        "SLEEP_POD_SUEDE", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_POLYESTER",
        "SLEEP_POD_NYLON", "SLEEP_POD_COTTON",

        "MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_SQUARE",
        "MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE",

        "PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL",

        "ROBOT_VACUUMING", "ROBOT_MOPPING", "ROBOT_DISHES",
        "ROBOT_LAUNDRY", "ROBOT_IRONING",

        "UV_VISOR_YELLOW", "UV_VISOR_AMBER", "UV_VISOR_ORANGE",
        "UV_VISOR_RED", "UV_VISOR_MAGENTA",

        "TRANSLATOR_SPACE_GRAY", "TRANSLATOR_ASTRO_BLACK",
        "TRANSLATOR_ECLIPSE_CHARCOAL", "TRANSLATOR_GRAPHITE_MIST",
        "TRANSLATOR_VOID_BLUE",

        "PANEL_1X2", "PANEL_2X2", "PANEL_1X4", "PANEL_2X4", "PANEL_4X4",

        "OXYGEN_SHAKE_MORNING_BREATH", "OXYGEN_SHAKE_EVENING_BREATH",
        "OXYGEN_SHAKE_MINT", "OXYGEN_SHAKE_CHOCOLATE",
        "OXYGEN_SHAKE_GARLIC",

        "SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA",
        "SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY",
        "SNACKPACK_RASPBERRY",
    ]

    # ---------- baseline OLD market-maker params (keep what worked) ----------
    OLD_PARAMS: Dict[str, Dict[str, float]] = {
        "SNACKPACK_CHOCOLATE": {"take": 2, "make": 2, "size": 3, "skew": 0.20},
        "SNACKPACK_VANILLA":   {"take": 2, "make": 2, "size": 3, "skew": 0.20},
        "SNACKPACK_PISTACHIO": {"take": 2, "make": 2, "size": 3, "skew": 0.20},
        "SNACKPACK_STRAWBERRY":{"take": 2, "make": 2, "size": 3, "skew": 0.20},
        "SNACKPACK_RASPBERRY": {"take": 2, "make": 2, "size": 3, "skew": 0.20},

        "PANEL_1X2": {"take": 2, "make": 2, "size": 3, "skew": 0.25},
        "PANEL_2X2": {"take": 2, "make": 2, "size": 3, "skew": 0.25},
        "PANEL_1X4": {"take": 2, "make": 2, "size": 3, "skew": 0.25},
        "PANEL_2X4": {"take": 3, "make": 3, "size": 2, "skew": 0.30},
        "PANEL_4X4": {"take": 3, "make": 3, "size": 2, "skew": 0.30},

        "PEBBLES_XS": {"take": 5, "make": 5, "size": 2, "skew": 0.50},
        "PEBBLES_S":  {"take": 5, "make": 5, "size": 2, "skew": 0.50},
        "PEBBLES_M":  {"take": 5, "make": 5, "size": 2, "skew": 0.50},
        "PEBBLES_L":  {"take": 5, "make": 5, "size": 2, "skew": 0.50},
        "PEBBLES_XL": {"take": 6, "make": 6, "size": 2, "skew": 0.60},

        "MICROCHIP_SQUARE": {"take": 5, "make": 5, "size": 1, "skew": 0.50},
        "ROBOT_DISHES":     {"take": 4, "make": 4, "size": 1, "skew": 0.50},
    }
    OLD_DEFAULT = {"take": 3, "make": 3, "size": 2, "skew": 0.30}

    # NEW_MM overrides for the few names where the prior new strat clearly helped
    NEW_MM: Dict[str, Dict[str, float]] = {
        "UV_VISOR_ORANGE":     {"take": 2, "make": 1, "size": 2, "skew": 0.55,
                                "min_spread": 4, "max_pos": 10},
        "SLEEP_POD_POLYESTER": {"take": 3, "make": 2, "size": 1, "skew": 0.60,
                                "min_spread": 4, "max_pos": 8},
        "SNACKPACK_VANILLA":   {"take": 2, "make": 1, "size": 3, "skew": 0.35,
                                "min_spread": 6, "max_pos": 10},
    }

    # Skip set: chronic bleeders from prior backtests where no robust directional
    # override clearly helps. Pebbles stay hard-disabled by request.
    SKIP = {
        # Ranked losers from the latest backtest (final pnl <= -1500 with drift driving it):
        "GALAXY_SOUNDS_PLANETARY_RINGS",   # -6247.7, drift -778
        "OXYGEN_SHAKE_MINT",               # -2029.8
        "SLEEP_POD_LAMB_WOOL",             # -1950.3
        "PEBBLES_XS",
        "PEBBLES_S",
        "PEBBLES_M",
        "PEBBLES_L",
        "PEBBLES_XL",

        # User-requested removals / hard-disabled products:
        "OXYGEN_SHAKE_EVENING_BREATH",
        "SLEEP_POD_POLYESTER",
        "UV_VISOR_MAGENTA",
        "UV_VISOR_YELLOW",
        "UV_VISOR_AMBER",
        "ROBOT_VACUUMING",
        "PANEL_2X2",

        # Disabled after IMC sim/backtester comparison: these created the main
        # loss tail in the official sim or had unstable backtester day profiles.
        "GALAXY_SOUNDS_BLACK_HOLES",
        "MICROCHIP_TRIANGLE",
        "OXYGEN_SHAKE_GARLIC",
        "PANEL_4X4",
        "ROBOT_LAUNDRY",
        "TRANSLATOR_ASTRO_BLACK",
        "TRANSLATOR_GRAPHITE_MIST",
        "TRANSLATOR_SPACE_GRAY",
        "UV_VISOR_ORANGE",
        "UV_VISOR_RED",
    }

    # PEBBLES basket fair: XS+S+M+L+XL ≈ 50_000.  Std ≈ 2.5.
    PEBBLES_BASKET = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
    PEBBLES_TARGET = 50_000.0
    # Per-pebble voluntary inventory cap. POSITION_LIMIT is 10 by exchange rules,
    # but we cap pebbles tighter to prevent the bag-hold pattern on PEBBLES_M
    # (v2: built short -6 from ts 47k to 59k as price kept rising, lost ~1,400).
    PEBBLE_POS_CAP = 5
    # Snackpack pair fair: CHOCOLATE + VANILLA ≈ 19_880. Std ≈ 11.
    SNACK_PAIR = ["SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA"]
    SNACK_TARGET = 19_880.0

    # EMA half-lives (in ticks) for trend-aware skew
    EMA_FAST_HL = 20
    EMA_SLOW_HL = 100

    # Directional overrides. direction=-1 means fade the EMA spread:
    # short when fast EMA is above slow EMA, long when it is below.
    DIRECTIONAL_TREND: Dict[str, Dict[str, float]] = {
        "GALAXY_SOUNDS_DARK_MATTER":
            {"fast": 50, "slow": 800, "threshold": 21, "exit": 10.5,
             "cap": 10, "direction": -1},
        "GALAXY_SOUNDS_SOLAR_FLAMES":
            {"fast": 50, "slow": 800, "threshold": 180, "exit": 90,
             "cap": 10, "direction": 1},
        "GALAXY_SOUNDS_SOLAR_WINDS":
            {"fast": 120, "slow": 500, "threshold": 233, "exit": 116.5,
             "cap": 10, "direction": -1},
        "SLEEP_POD_COTTON":
            {"fast": 20, "slow": 200, "threshold": 120, "exit": 60,
             "cap": 10, "direction": 1},
        "SLEEP_POD_SUEDE":
            {"fast": 10, "slow": 100, "threshold": 120, "exit": 60,
             "cap": 10, "direction": -1},
        "MICROCHIP_RECTANGLE":
            {"fast": 120, "slow": 200, "threshold": 3, "exit": 1.5,
             "cap": 10, "direction": -1},
        "MICROCHIP_CIRCLE":
            {"fast": 120, "slow": 500, "threshold": 8, "exit": 4,
             "cap": 10, "direction": 1},
        "MICROCHIP_OVAL":
            {"fast": 5, "slow": 800, "threshold": 144, "exit": 72,
             "cap": 10, "direction": 1},
        "MICROCHIP_SQUARE":
            {"fast": 20, "slow": 100, "threshold": 144, "exit": 72,
             "cap": 10, "direction": 1},
        "ROBOT_MOPPING":
            {"fast": 20, "slow": 200, "threshold": 34, "exit": 17,
             "cap": 10, "direction": 1},
        "ROBOT_IRONING":
            {"fast": 5, "slow": 800, "threshold": 180, "exit": 90,
             "cap": 10, "direction": 1},
        "TRANSLATOR_ECLIPSE_CHARCOAL":
            {"fast": 50, "slow": 800, "threshold": 377, "exit": 188.5,
             "cap": 10, "direction": -1},
        "TRANSLATOR_VOID_BLUE":
            {"fast": 5, "slow": 800, "threshold": 233, "exit": 116.5,
             "cap": 10, "direction": -1},
        "PANEL_1X4":
            {"fast": 10, "slow": 100, "threshold": 34, "exit": 17,
             "cap": 10, "direction": 1},
        "PANEL_2X4":
            {"fast": 120, "slow": 800, "threshold": 144, "exit": 72,
             "cap": 10, "direction": -1},
        "OXYGEN_SHAKE_MORNING_BREATH":
            {"fast": 80, "slow": 300, "threshold": 1, "exit": 0.5,
             "cap": 10, "direction": 1},
        "PANEL_1X2":
            {"fast": 120, "slow": 500, "threshold": 144, "exit": 72,
             "cap": 10, "direction": -1},
        "ROBOT_DISHES":    {"fast": 120, "slow": 500, "threshold": 89, "exit": 44.5,
                            "cap": 10, "direction": -1},
        "SLEEP_POD_NYLON":
            {"fast": 120, "slow": 800, "threshold": 2, "exit": 1,
             "cap": 10, "direction": -1},
        "SNACKPACK_RASPBERRY":
            {"fast": 5, "slow": 800, "threshold": 180, "exit": 90,
             "cap": 10, "direction": -1},
        "SNACKPACK_CHOCOLATE":
            {"fast": 120, "slow": 800, "threshold": 1, "exit": 0.5,
             "cap": 10, "direction": -1},
        "SNACKPACK_VANILLA":
            {"fast": 120, "slow": 800, "threshold": 5, "exit": 2.5,
             "cap": 10, "direction": -1},
        "SNACKPACK_PISTACHIO":
            {"fast": 120, "slow": 800, "threshold": 8, "exit": 4,
             "cap": 10, "direction": -1},
        "SNACKPACK_STRAWBERRY":
            {"fast": 120, "slow": 200, "threshold": 13, "exit": 6.5,
             "cap": 10, "direction": -1},
        "TRANSLATOR_SPACE_GRAY":
            {"fast": 20, "slow": 100, "threshold": 34, "exit": 17,
             "cap": 10, "direction": 1},
    }

    # --------- helpers ---------
    def _book(self, depth: OrderDepth) -> Optional[Tuple[int, int, int, int, float, int]]:
        if not depth.buy_orders or not depth.sell_orders:
            return None
        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        bid_vol = depth.buy_orders[best_bid]
        ask_vol = -depth.sell_orders[best_ask]
        spread = best_ask - best_bid
        if bid_vol + ask_vol > 0:
            fair = (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
        else:
            fair = (best_bid + best_ask) / 2
        return best_bid, best_ask, bid_vol, ask_vol, fair, spread

    def _mid(self, depth: OrderDepth) -> Optional[float]:
        if not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders) + min(depth.sell_orders)) / 2

    @staticmethod
    def _ema_alpha(half_life: float) -> float:
        return 1.0 - math.exp(-math.log(2) / half_life)

    # --------- basket fair-value anchors ---------
    def _pebble_anchor(self, state: TradingState) -> Optional[float]:
        """Returns target XL for current basket state, or None if missing."""
        mids = []
        for p in self.PEBBLES_BASKET:
            if p not in state.order_depths:
                return None
            m = self._mid(state.order_depths[p])
            if m is None:
                return None
            mids.append(m)
        # We just want the basket residual: how far the sum is from PEBBLES_TARGET
        return self.PEBBLES_TARGET - sum(mids[:-1])  # this is what XL "should" be by identity

    def _pebble_basket_residual(self, state: TradingState) -> Optional[float]:
        """Residual = current basket sum - target.  >0 means basket rich, <0 means cheap."""
        s = 0.0
        for p in self.PEBBLES_BASKET:
            m = self._mid(state.order_depths.get(p, None)) if p in state.order_depths else None
            if m is None:
                return None
            s += m
        return s - self.PEBBLES_TARGET

    def _snack_pair_residual(self, state: TradingState) -> Optional[float]:
        s = 0.0
        for p in self.SNACK_PAIR:
            if p not in state.order_depths:
                return None
            m = self._mid(state.order_depths[p])
            if m is None:
                return None
            s += m
        return s - self.SNACK_TARGET

    # --------- core market-maker engine (with trend skew + basket overlay) ---------
    def _mm_trade(
        self,
        product: str,
        depth: OrderDepth,
        position: int,
        params: Dict[str, float],
        memory: Dict[str, float],
        fair_override: Optional[float] = None,
        extra_skew: float = 0.0,
        position_cap: Optional[int] = None,
    ) -> List[Order]:
        if not depth.buy_orders or not depth.sell_orders:
            return []

        # If a tighter cap is given, use min(POSITION_LIMIT, cap). Otherwise full limit.
        # The exchange enforces POSITION_LIMIT; the cap is voluntary risk control.
        limit = self.POSITION_LIMIT
        effective_cap = limit if position_cap is None else min(limit, int(position_cap))
        take_edge = params["take"]
        make_edge = params["make"]
        quote_size = int(params["size"])
        skew = params["skew"]

        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        mid = (best_bid + best_ask) / 2

        # Update EMAs
        af = self._ema_alpha(self.EMA_FAST_HL)
        as_ = self._ema_alpha(self.EMA_SLOW_HL)
        ema_f = memory.get(f"{product}_ema_f", mid)
        ema_s = memory.get(f"{product}_ema_s", mid)
        ema_f = ema_f + af * (mid - ema_f)
        ema_s = ema_s + as_ * (mid - ema_s)
        memory[f"{product}_ema_f"] = ema_f
        memory[f"{product}_ema_s"] = ema_s

        # Trend signal: positive = uptrend
        trend = ema_f - ema_s
        # Convert to a small price adjustment. Cap to keep this from going wild on big drifts.
        # Empirically std of trend on most products is ~5-15.  A skew of 0.3 * trend caps fine.
        trend_skew = 0.30 * trend
        # Clip so we never push fair more than 1 tick beyond mid from this term
        trend_skew = max(-3.0, min(3.0, trend_skew))

        if fair_override is not None:
            fair = fair_override
        else:
            fair = mid

        # position skew + trend skew + basket extra skew
        fair_adj = fair - skew * position + trend_skew + extra_skew

        # Voluntary cap: we can grow inventory only up to effective_cap, even though
        # the hard exchange limit may be larger. This prevents the +6 / -6 build-and-bag
        # pattern that hurt PEBBLES_M (-1,413) in v2.
        buy_cap = max(0, effective_cap - position)
        sell_cap = max(0, effective_cap + position)
        orders: List[Order] = []

        # Take cheap asks
        for ask in sorted(depth.sell_orders):
            if buy_cap <= 0: break
            ask_volume = -depth.sell_orders[ask]
            if fair_adj - ask >= take_edge:
                qty = min(ask_volume, buy_cap)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    buy_cap -= qty
            else:
                break

        # Take expensive bids
        for bid in sorted(depth.buy_orders, reverse=True):
            if sell_cap <= 0: break
            bid_volume = depth.buy_orders[bid]
            if bid - fair_adj >= take_edge:
                qty = min(bid_volume, sell_cap)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    sell_cap -= qty
            else:
                break

        # Make inside spread
        max_bid = math.floor(fair_adj - make_edge)
        my_bid = min(best_bid + 1, max_bid, best_ask - 1)
        if buy_cap > 0 and my_bid > best_bid:
            orders.append(Order(product, my_bid, min(quote_size, buy_cap)))

        min_ask = math.ceil(fair_adj + make_edge)
        my_ask = max(best_ask - 1, min_ask, best_bid + 1)
        if sell_cap > 0 and my_ask < best_ask:
            orders.append(Order(product, my_ask, -min(quote_size, sell_cap)))

        return orders

    def _directional_trend_trade(
        self,
        product: str,
        depth: OrderDepth,
        position: int,
        cfg: Dict[str, float],
        memory: Dict[str, float],
    ) -> List[Order]:
        if not depth.buy_orders or not depth.sell_orders:
            return []

        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        mid = (best_bid + best_ask) / 2

        af = self._ema_alpha(cfg["fast"])
        as_ = self._ema_alpha(cfg["slow"])
        kf = f"{product}_rt_ema_f"
        ks = f"{product}_rt_ema_s"
        ema_f = memory.get(kf, mid)
        ema_s = memory.get(ks, mid)
        ema_f = ema_f + af * (mid - ema_f)
        ema_s = ema_s + as_ * (mid - ema_s)
        memory[kf] = ema_f
        memory[ks] = ema_s

        signal = cfg["direction"] * (ema_f - ema_s)
        cap = min(self.POSITION_LIMIT, int(cfg["cap"]))

        if signal > cfg["threshold"]:
            target = cap
        elif signal < -cfg["threshold"]:
            target = -cap
        elif abs(signal) < cfg["exit"]:
            target = 0
        else:
            target = position

        delta = target - position
        orders: List[Order] = []

        if delta > 0:
            remaining = min(delta, self.POSITION_LIMIT - position)
            for ask in sorted(depth.sell_orders):
                if remaining <= 0:
                    break
                qty = min(-depth.sell_orders[ask], remaining)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    remaining -= qty
        elif delta < 0:
            remaining = min(-delta, self.POSITION_LIMIT + position)
            for bid in sorted(depth.buy_orders, reverse=True):
                if remaining <= 0:
                    break
                qty = min(depth.buy_orders[bid], remaining)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    remaining -= qty

        return orders

    # --------- main entry ---------
    def run(self, state: TradingState):
        # Load memory
        try:
            memory = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            memory = {}

        result: Dict[str, List[Order]] = {}

        # Compute basket residuals once
        peb_resid = self._pebble_basket_residual(state)   # >0: pebbles rich; <0: cheap
        snack_resid = self._snack_pair_residual(state)    # >0: pair rich

        for product, depth in state.order_depths.items():
            pos = state.position.get(product, 0)

            try:
                if product in self.SKIP:
                    result[product] = []
                    continue

                if product in self.DIRECTIONAL_TREND:
                    result[product] = self._directional_trend_trade(
                        product, depth, pos, self.DIRECTIONAL_TREND[product], memory
                    )
                    continue

                # Decide which params to use
                if product in self.NEW_MM:
                    params = dict(self.NEW_MM[product])
                else:
                    params = dict(self.OLD_PARAMS.get(product, self.OLD_DEFAULT))

                # Compute fair-value override + extra skew from baskets
                fair_override = None
                extra = 0.0

                if product in self.PEBBLES_BASKET and peb_resid is not None:
                    # If basket residual is +R, the average pebble mid is +R/5 too rich.
                    # Push our fair down by R/5 so we lean to sell. Same logic if R<0.
                    extra -= peb_resid / 5.0
                if product in self.SNACK_PAIR and snack_resid is not None:
                    # Pair sum = 19880; if residual +R, pair is +R total too rich. Each leg
                    # should drop by R/2.
                    extra -= snack_resid / 2.0

                # NEW_MM has its own min_spread gate; honor it lightly
                if product in self.NEW_MM:
                    cfg = self.NEW_MM[product]
                    bb, ba, _, _, _, sp = self._book(depth) or (0, 0, 0, 0, 0, 0)
                    if sp < cfg.get("min_spread", 0):
                        result[product] = []
                        continue

                # Pebbles get a tighter voluntary inventory cap (±5 instead of ±10)
                # because v2 showed they bag-hold during trends and lose hard.
                # Other products keep the full ±10 limit.
                pos_cap = self.PEBBLE_POS_CAP if product in self.PEBBLES_BASKET else None

                result[product] = self._mm_trade(
                    product, depth, pos, params, memory,
                    fair_override=fair_override,
                    extra_skew=extra,
                    position_cap=pos_cap,
                )

            except Exception as e:
                print(f"Error trading {product}: {e}")
                result[product] = []

        # Persist memory (keep size small — only EMAs)
        try:
            traderData = json.dumps(memory)
        except Exception:
            traderData = ""

        return result, 0, traderData