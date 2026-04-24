from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional
from math import ceil, erf, log, sqrt
import json


POSITION_LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
}

VEV_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV_LIMIT = 300
for _strike in VEV_STRIKES:
    POSITION_LIMITS[f"VEV_{_strike}"] = VEV_LIMIT


DAY_LEN_DEFAULT = 1_000_000.0
OPEN_TTE_CANDIDATES = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

HG_CONFIG = {
    "anchor": 9991.0,
    "ema_alpha": 0.060,
    "ema_weight": 0.72,
    "micro_weight": 2.40,
    "micro_cap": 6.0,
    "take_edge": 5.0,
    "post_edge": 4.0,
    "skew_div": 40.0,
    "max_take": 35,
    "max_post": 28,
}

VE_CONFIG = {
    "anchor": 5250.0,
    "ema_alpha": 0.075,
    "ema_weight": 0.70,
    "micro_weight": 0.55,
    "micro_cap": 2.5,
    "take_edge": 99.0,
    "post_edge": 99.0,
    "skew_div": 55.0,
    "max_take": 0,
    "max_post": 0,
}

VEV_SIGMA_DEFAULT = 0.0125
VEV_SIGMA_FLOOR = 0.0100
VEV_SIGMA_CEIL = 0.0175
VEV_SIGMA_ALPHA = 0.12
VEV_BIAS_ALPHA = 0.04
VEV_BIAS_CAP = 4.0
VEV_SIGMA_OFFSETS = {
    5000: 0.00017,
    5100: 0.00004,
    5200: 0.00018,
    5300: 0.00032,
    5400: -0.00048,
    5500: 0.00035,
}
VEV_CALIB_STRIKES = [5100, 5200, 5300, 5400]
VEV_ACTIVE_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]
VEV_POST_STRIKES = {5000, 5100, 5200, 5300, 5400, 5500}
VEV_MIN_FAIR = 1.0
VEV_MAX_TAKE_PER_PRODUCT = 36
VEV_MAX_POST_SIZE = 10
VEV_MAX_OPTION_DELTA = 175.0
VEV_OPTION_INV_PENALTY = 0.025
VEV_PORTFOLIO_TILT = 0.020
VEV_DEV_EMA_ALPHA = 0.055
VEV_ABSDEV_EMA_ALPHA = 0.020
VEV_SIGNAL_FLOOR = 0.45
VEV_SIGNAL_BAND_MULT = 1.8
VEV_SIGNAL_TILT = 0.85
VEV_SIGNAL_CLOSE_MULT = 2.6
VEV_SIGNAL_CLOSE_SIZE = 18

VE_HEDGE_CROSS_SLICE = 24
VE_HEDGE_DEADBAND = 8
VE_HEDGE_URGENT_GAP = 35
VE_HEDGE_REST_SIZE = 12


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_call(spot: float, strike: float, tte: float, sigma: float) -> float:
    if spot <= 0.0:
        return 0.0
    if tte <= 0.0 or sigma <= 0.0:
        return max(0.0, spot - strike)

    sigma_t = sigma * sqrt(tte)
    d1 = (log(spot / strike) + 0.5 * sigma * sigma * tte) / sigma_t
    d2 = d1 - sigma_t
    return spot * _norm_cdf(d1) - strike * _norm_cdf(d2)


def bs_delta(spot: float, strike: float, tte: float, sigma: float) -> float:
    if spot <= 0.0:
        return 0.0
    if tte <= 0.0 or sigma <= 0.0:
        return 1.0 if spot > strike else 0.0

    sigma_t = sigma * sqrt(tte)
    d1 = (log(spot / strike) + 0.5 * sigma * sigma * tte) / sigma_t
    return _norm_cdf(d1)


def implied_vol(price: float, spot: float, strike: float, tte: float) -> float:
    intrinsic = max(0.0, spot - strike)
    if price <= intrinsic + 1e-9 or spot <= 0.0 or tte <= 0.0:
        return 0.0

    lo, hi = 1e-4, 1.0
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if bs_call(spot, strike, tte, mid) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        trader_data = self._load_trader_data(state.traderData)
        day_len = self._update_day_tracking(trader_data, state.timestamp)

        ve_order_depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
        ve_mid = self._wall_mid(ve_order_depth) if ve_order_depth else None

        if trader_data.get("open_tte") is None and ve_mid is not None:
            trader_data["open_tte"] = self._infer_open_tte(
                state.order_depths,
                ve_mid,
                state.timestamp,
                day_len,
                trader_data.get("sigma", VEV_SIGMA_DEFAULT),
            )

        if trader_data.get("open_tte") is None:
            trader_data["open_tte"] = 5.0

        tte = max(1e-4, trader_data["open_tte"] - state.timestamp / day_len)

        result: Dict[str, List[Order]] = {}
        ema_store = trader_data.setdefault("ema", {})

        underlying_fairs: Dict[str, float] = {}
        for product, config in (
            ("HYDROGEL_PACK", HG_CONFIG),
            ("VELVETFRUIT_EXTRACT", VE_CONFIG),
        ):
            order_depth = state.order_depths.get(product)
            if order_depth is None:
                continue
            fair = self._update_underlying_fair(
                product, order_depth, ema_store, config
            )
            underlying_fairs[product] = fair

        sigma = self._update_sigma(
            trader_data,
            state.order_depths,
            ve_mid,
            state.timestamp,
            day_len,
        )

        ve_spot_ref = ve_mid
        if ve_mid is not None and "VELVETFRUIT_EXTRACT" in underlying_fairs:
            ve_fair = underlying_fairs["VELVETFRUIT_EXTRACT"]
            ve_spot_ref = ve_mid + 0.35 * (ve_fair - ve_mid)

        option_orders: Dict[str, List[Order]] = {}
        projected_option_delta = 0.0
        if ve_spot_ref is not None and tte > 0.01:
            option_orders, projected_option_delta = self._trade_vouchers(
                state,
                trader_data,
                ve_spot_ref,
                tte,
                sigma,
            )
        else:
            for strike in VEV_STRIKES:
                option_orders[f"VEV_{strike}"] = []

        result.update(option_orders)

        if ve_order_depth is not None:
            ve_position = state.position.get("VELVETFRUIT_EXTRACT", 0)
            hedge_target = self._clamp(
                -int(round(projected_option_delta)),
                -POSITION_LIMITS["VELVETFRUIT_EXTRACT"],
                POSITION_LIMITS["VELVETFRUIT_EXTRACT"],
            )
            result["VELVETFRUIT_EXTRACT"] = self._hedge_ve(
                "VELVETFRUIT_EXTRACT",
                ve_order_depth,
                ve_position,
                hedge_target,
            )

        hg_order_depth = state.order_depths.get("HYDROGEL_PACK")
        if hg_order_depth is not None:
            hg_position = state.position.get("HYDROGEL_PACK", 0)
            result["HYDROGEL_PACK"] = self._market_make_underlying(
                "HYDROGEL_PACK",
                hg_order_depth,
                hg_position,
                POSITION_LIMITS["HYDROGEL_PACK"],
                underlying_fairs.get("HYDROGEL_PACK", HG_CONFIG["anchor"]),
                0,
                HG_CONFIG,
            )

        return result, 0, json.dumps(trader_data, separators=(",", ":"))

    def _load_trader_data(self, raw: str) -> Dict:
        if not raw:
            return {"sigma": VEV_SIGMA_DEFAULT}
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                return {"sigma": VEV_SIGMA_DEFAULT}
            if "sigma" not in data:
                data["sigma"] = VEV_SIGMA_DEFAULT
            return data
        except Exception:
            return {"sigma": VEV_SIGMA_DEFAULT}

    def _update_day_tracking(self, trader_data: Dict, timestamp: int) -> float:
        prev_timestamp = trader_data.get("prev_timestamp")
        day_len = float(trader_data.get("day_len", DAY_LEN_DEFAULT))

        if prev_timestamp is not None and timestamp < prev_timestamp:
            observed_day_len = float(prev_timestamp + 100)
            if observed_day_len > 100_000:
                day_len = observed_day_len
            trader_data["open_tte"] = max(1.0, float(trader_data.get("open_tte", 5.0)) - 1.0)

        trader_data["prev_timestamp"] = timestamp
        trader_data["day_len"] = day_len
        return day_len

    def _infer_open_tte(
        self,
        order_depths: Dict[str, OrderDepth],
        ve_mid: float,
        timestamp: int,
        day_len: float,
        sigma_hint: float,
    ) -> float:
        best_open_tte = 5.0
        best_score = float("inf")

        for open_tte in OPEN_TTE_CANDIDATES:
            tte = open_tte - timestamp / day_len
            if tte <= 0.25:
                continue

            ivs: List[float] = []
            for strike in VEV_CALIB_STRIKES:
                option_mid = self._mid(order_depths.get(f"VEV_{strike}"))
                if option_mid is None:
                    continue
                iv = implied_vol(option_mid, ve_mid, strike, tte)
                if VEV_SIGMA_FLOOR <= iv <= 0.03:
                    ivs.append(iv)

            if len(ivs) < 2:
                continue

            mean_iv = sum(ivs) / len(ivs)
            variance = sum((iv - mean_iv) ** 2 for iv in ivs) / len(ivs)
            score = variance + 5.0 * abs(mean_iv - sigma_hint)
            if score < best_score:
                best_score = score
                best_open_tte = open_tte

        return best_open_tte

    def _update_underlying_fair(
        self,
        product: str,
        order_depth: OrderDepth,
        ema_store: Dict[str, float],
        config: Dict[str, float],
    ) -> float:
        ref_mid = self._wall_mid(order_depth)
        if ref_mid is None:
            return config["anchor"]

        previous_ema = float(ema_store.get(product, ref_mid))
        ema = (1.0 - config["ema_alpha"]) * previous_ema + config["ema_alpha"] * ref_mid
        ema_store[product] = ema

        micro = self._microprice(order_depth)
        if micro is None:
            micro = ref_mid

        fair = config["ema_weight"] * ema + (1.0 - config["ema_weight"]) * config["anchor"]
        micro_shift = self._clamp(
            config["micro_weight"] * (micro - ref_mid),
            -config["micro_cap"],
            config["micro_cap"],
        )
        return fair + micro_shift

    def _update_sigma(
        self,
        trader_data: Dict,
        order_depths: Dict[str, OrderDepth],
        ve_mid: Optional[float],
        timestamp: int,
        day_len: float,
    ) -> float:
        sigma = float(trader_data.get("sigma", VEV_SIGMA_DEFAULT))
        open_tte = float(trader_data.get("open_tte", 5.0))
        tte = max(1e-4, open_tte - timestamp / day_len)

        if ve_mid is None or tte <= 0.01:
            trader_data["sigma"] = sigma
            return sigma

        ivs: List[float] = []
        for strike in VEV_CALIB_STRIKES:
            option_mid = self._mid(order_depths.get(f"VEV_{strike}"))
            if option_mid is None:
                continue
            iv = implied_vol(option_mid, ve_mid, strike, tte)
            if VEV_SIGMA_FLOOR <= iv <= VEV_SIGMA_CEIL:
                ivs.append(iv)

        if ivs:
            ivs.sort()
            robust_iv = ivs[len(ivs) // 2]
            sigma = (1.0 - VEV_SIGMA_ALPHA) * sigma + VEV_SIGMA_ALPHA * robust_iv

        sigma = self._clamp(sigma, VEV_SIGMA_FLOOR, VEV_SIGMA_CEIL)
        trader_data["sigma"] = sigma
        return sigma

def _trade_vouchers(
    self,
    state: TradingState,
    trader_data: Dict,
    spot_ref: float,
    tte: float,
    sigma: float,
) -> Tuple[Dict[str, List[Order]], float]:

    biases = trader_data.setdefault(
        "vev_bias",
        {str(strike): 0.0 for strike in VEV_STRIKES},
    )

    projected_option_delta = 0.0
    orders: Dict[str, List[Order]] = {f"VEV_{strike}": [] for strike in VEV_STRIKES}

    for strike in VEV_STRIKES:
        product = f"VEV_{strike}"
        order_depth = state.order_depths.get(product)
        position = state.position.get(product, 0)

        if order_depth is None:
            continue

        # --- Black-Scholes ---
        strike_sigma = self._strike_sigma(strike, sigma)
        base_fair = bs_call(spot_ref, strike, tte, strike_sigma)
        delta = bs_delta(spot_ref, strike, tte, strike_sigma)
        projected_option_delta += position * delta

        mid = self._mid(order_depth)
        if mid is None:
            continue

        # --- Bias learning (same as before, but safer bounds) ---
        bias_key = str(strike)
        observed_bias = mid - base_fair
        biases[bias_key] = self._clamp(
            (1.0 - VEV_BIAS_ALPHA) * biases[bias_key] + VEV_BIAS_ALPHA * observed_bias,
            -3.0, 3.0,
        )

        # --- 🔥 NEW: Vol skew penalty ---
        moneyness = (spot_ref - strike) / spot_ref

        skew_penalty = 0.0
        if moneyness < 0:  # OTM calls
            skew_penalty = -abs(moneyness) * 45.0  # key fix

        fair = max(0.0, base_fair + biases[bias_key] + skew_penalty)

        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None

        if best_bid is None or best_ask is None:
            continue

        spread = best_ask - best_bid

        # --- 🔥 NEW: adaptive edge threshold ---
        take_edge = max(1.0, 0.5 * spread)

        if strike >= 5300:
            take_edge *= 1.8   # less aggressive for weak strikes

        # --- Position cap (new) ---
        pos_limit = 300
        if strike >= 5300:
            pos_limit = 120

        # =========================
        # BUY LOGIC
        # =========================
        for ask, vol in sorted(order_depth.sell_orders.items()):
            if position >= pos_limit:
                break

            edge = fair - ask
            if edge < take_edge:
                break

            size = min(-vol, pos_limit - position, 25)
            if size <= 0:
                continue

            orders[product].append(Order(product, ask, size))
            position += size

        # =========================
        # SELL LOGIC
        # =========================
        for bid, vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if position <= -pos_limit:
                break

            edge = bid - fair
            if edge < take_edge:
                break

            size = min(vol, pos_limit + position, 25)
            if size <= 0:
                continue

            orders[product].append(Order(product, bid, -size))
            position -= size

    return orders, projected_option_delta

    def _market_make_underlying(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
        limit: int,
        fair: float,
        target_position: int,
        config: Dict[str, float],
    ) -> List[Order]:
        orders: List[Order] = []
        running_position = position
        buy_room = limit - running_position
        sell_room = limit + running_position

        bought = 0
        if order_depth.sell_orders:
            for ask in sorted(order_depth.sell_orders):
                if ask > fair - config["take_edge"] or bought >= config["max_take"] or buy_room <= 0:
                    break
                qty = min(abs(order_depth.sell_orders[ask]), buy_room, config["max_take"] - bought)
                if qty <= 0:
                    continue
                orders.append(Order(product, int(ask), qty))
                bought += qty
                running_position += qty
                buy_room -= qty

        sold = 0
        if order_depth.buy_orders:
            for bid in sorted(order_depth.buy_orders, reverse=True):
                if bid < fair + config["take_edge"] or sold >= config["max_take"] or sell_room <= 0:
                    break
                qty = min(order_depth.buy_orders[bid], sell_room, config["max_take"] - sold)
                if qty <= 0:
                    continue
                orders.append(Order(product, int(bid), -qty))
                sold += qty
                running_position -= qty
                sell_room -= qty

        imbalance = running_position - target_position
        skew = imbalance / config["skew_div"]
        bid_px = int(fair - config["post_edge"] - skew)
        ask_px = int(ceil(fair + config["post_edge"] - skew))

        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
        if best_ask is not None:
            bid_px = min(bid_px, best_ask - 1)
        if best_bid is not None:
            ask_px = max(ask_px, best_bid + 1)

        buy_size = min(config["max_post"], buy_room)
        sell_size = min(config["max_post"], sell_room)
        if imbalance < -15:
            buy_size = min(buy_room, config["max_post"] + int(abs(imbalance) / 5))
            sell_size = min(sell_room, max(0, config["max_post"] // 2))
        elif imbalance > 15:
            sell_size = min(sell_room, config["max_post"] + int(abs(imbalance) / 5))
            buy_size = min(buy_room, max(0, config["max_post"] // 2))

        if buy_size > 0 and bid_px > 0:
            orders.append(Order(product, bid_px, buy_size))
        if sell_size > 0 and ask_px > 0:
            orders.append(Order(product, ask_px, -sell_size))

        return orders

    def _hedge_ve(
        self,
        product: str,
        order_depth: OrderDepth,
        current_position: int,
        target_position: int,
    ) -> List[Order]:
        orders: List[Order] = []
        delta_gap = target_position - current_position
        if abs(delta_gap) <= VE_HEDGE_DEADBAND:
            return orders

        limit = POSITION_LIMITS[product]
        remaining = abs(delta_gap)
        assumed_position = current_position

        if delta_gap > 0:
            cross_remaining = 0
            if remaining >= VE_HEDGE_URGENT_GAP and order_depth.sell_orders:
                cross_remaining = min(
                    VE_HEDGE_CROSS_SLICE,
                    remaining - VE_HEDGE_DEADBAND,
                    limit - assumed_position,
                )
            for ask in sorted(order_depth.sell_orders):
                if cross_remaining <= 0:
                    break
                qty = min(
                    abs(order_depth.sell_orders[ask]),
                    cross_remaining,
                    limit - assumed_position,
                )
                if qty <= 0:
                    continue
                orders.append(Order(product, int(ask), qty))
                cross_remaining -= qty
                remaining -= qty
                assumed_position += qty
            rest_qty = min(remaining, VE_HEDGE_REST_SIZE, limit - assumed_position)
            if rest_qty > 0:
                passive_buy = self._best_passive_buy(order_depth)
                if passive_buy is not None:
                    orders.append(Order(product, passive_buy, rest_qty))

        else:
            cross_remaining = 0
            if remaining >= VE_HEDGE_URGENT_GAP and order_depth.buy_orders:
                cross_remaining = min(
                    VE_HEDGE_CROSS_SLICE,
                    remaining - VE_HEDGE_DEADBAND,
                    limit + assumed_position,
                )
            for bid in sorted(order_depth.buy_orders, reverse=True):
                if cross_remaining <= 0:
                    break
                qty = min(
                    order_depth.buy_orders[bid],
                    cross_remaining,
                    limit + assumed_position,
                )
                if qty <= 0:
                    continue
                orders.append(Order(product, int(bid), -qty))
                cross_remaining -= qty
                remaining -= qty
                assumed_position -= qty
            rest_qty = min(remaining, VE_HEDGE_REST_SIZE, limit + assumed_position)
            if rest_qty > 0:
                passive_sell = self._best_passive_sell(order_depth)
                if passive_sell is not None:
                    orders.append(Order(product, passive_sell, -rest_qty))

        return orders

    @staticmethod
    def _mid(order_depth: Optional[OrderDepth]) -> Optional[float]:
        if order_depth is None:
            return None
        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
        if best_bid is not None and best_ask is not None:
            return 0.5 * (best_bid + best_ask)
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)
        return None

    @staticmethod
    def _wall_mid(order_depth: Optional[OrderDepth]) -> Optional[float]:
        if order_depth is None:
            return None
        bid_wall = min(order_depth.buy_orders) if order_depth.buy_orders else None
        ask_wall = max(order_depth.sell_orders) if order_depth.sell_orders else None
        if bid_wall is not None and ask_wall is not None:
            return 0.5 * (bid_wall + ask_wall)
        return Trader._mid(order_depth)

    @staticmethod
    def _microprice(order_depth: Optional[OrderDepth]) -> Optional[float]:
        if order_depth is None or not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        bid_volume = float(order_depth.buy_orders[best_bid])
        ask_volume = float(abs(order_depth.sell_orders[best_ask]))
        total = bid_volume + ask_volume
        if total <= 0.0:
            return 0.5 * (best_bid + best_ask)
        return (best_ask * bid_volume + best_bid * ask_volume) / total

    @staticmethod
    def _spread(order_depth: Optional[OrderDepth]) -> Optional[float]:
        if order_depth is None or not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        return float(min(order_depth.sell_orders) - max(order_depth.buy_orders))

    @staticmethod
    def _best_passive_buy(order_depth: Optional[OrderDepth]) -> Optional[int]:
        if order_depth is None:
            return None
        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
        if best_bid is not None and best_ask is not None:
            return max(1, min(best_bid + 1, best_ask - 1))
        if best_bid is not None:
            return max(1, best_bid)
        if best_ask is not None:
            return max(1, best_ask - 1)
        return None

    @staticmethod
    def _best_passive_sell(order_depth: Optional[OrderDepth]) -> Optional[int]:
        if order_depth is None:
            return None
        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
        if best_bid is not None and best_ask is not None:
            return max(1, max(best_ask - 1, best_bid + 1))
        if best_ask is not None:
            return max(1, best_ask)
        if best_bid is not None:
            return max(1, best_bid + 1)
        return None

    @staticmethod
    def _delta_room_buy(current_delta: float, contract_delta: float) -> int:
        if contract_delta <= 1e-9:
            return VEV_LIMIT
        room = (VEV_MAX_OPTION_DELTA - current_delta) / contract_delta
        return max(0, int(room))

    @staticmethod
    def _delta_room_sell(current_delta: float, contract_delta: float) -> int:
        if contract_delta <= 1e-9:
            return VEV_LIMIT
        room = (current_delta + VEV_MAX_OPTION_DELTA) / contract_delta
        return max(0, int(room))

    def _strike_sigma(self, strike: int, sigma: float) -> float:
        return self._clamp(
            sigma + VEV_SIGMA_OFFSETS.get(strike, 0.0),
            VEV_SIGMA_FLOOR,
            VEV_SIGMA_CEIL,
        )

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))