from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import json


POSITION_LIMITS = {
    "INTARIAN_PEPPER_ROOT": 80,
    "ASH_COATED_OSMIUM":    80,
}

# Pepper
PEPPER_SELL_FRACTION = 0.02
PEPPER_BUY_BUFFER    = 8

# Osmium
OSM_ANCHOR           = 10000
OSM_FV_CLAMP         = 5
OSM_EMA_ALPHA        = 0.15
OSM_SKEW_FACTOR      = 3 

DEFAULT_DAY_LEN      = 1_000_000


class Trader:

    def bid(self):
        return 0

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        prev_ts = td.get("prev_ts", -1)
        if prev_ts > 50_000 and state.timestamp < prev_ts:
            td["day_len"] = prev_ts + 100
        td["prev_ts"] = state.timestamp
        day_len = td.get("day_len", DEFAULT_DAY_LEN)

        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            pos   = state.position.get(product, 0)
            limit = POSITION_LIMITS[product]

            if product == "INTARIAN_PEPPER_ROOT":
                result[product] = self._pepper(
                    od, pos, limit, state.timestamp, day_len
                )
            elif product == "ASH_COATED_OSMIUM":
                result[product], td = self._osmium(od, pos, limit, td)

        return result, 0, json.dumps(td)

    # ──────────────────────────────────────────────────────
    #  INTARIAN_PEPPER_ROOT — trend follow, buy-and-hold
    # ──────────────────────────────────────────────────────
    def _pepper(
        self,
        od: OrderDepth,
        pos: int,
        limit: int,
        ts: int,
        day_len: int,
    ) -> List[Order]:
        orders: List[Order] = []
        if not od.buy_orders and not od.sell_orders:
            return orders

        best_bid = max(od.buy_orders.keys())  if od.buy_orders  else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        if best_bid and best_ask:
            mid = (best_bid + best_ask) / 2
        elif best_bid:
            mid = float(best_bid)
        else:
            mid = float(best_ask)

        sell_window = int(day_len * PEPPER_SELL_FRACTION)
        near_end    = ts >= day_len - sell_window

        # END OF DAY: flatten position
        if near_end and pos > 0 and od.buy_orders:
            remaining = pos
            for bid in sorted(od.buy_orders.keys(), reverse=True):
                if remaining <= 0:
                    break
                qty = min(od.buy_orders[bid], remaining)
                if qty > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", bid, -qty))
                    remaining -= qty
            return orders

        # ACCUMULATE: take any ask ≤ mid + buffer
        # FIX: buffer raised to 8 so we actually cross the spread
        buy_budget = limit - pos
        threshold  = mid + PEPPER_BUY_BUFFER
        if buy_budget > 0 and od.sell_orders:
            for ask in sorted(od.sell_orders.keys()):
                if buy_budget <= 0 or ask > threshold:
                    break
                qty = min(abs(od.sell_orders[ask]), buy_budget)
                if qty > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", ask, qty))
                    buy_budget -= qty

        # OVERBID: sit 1 tick above best bid, below fair value
        if buy_budget > 0 and od.buy_orders:
            target_bid = None
            for bp in sorted(od.buy_orders.keys(), reverse=True):
                overbid = bp + 1
                if overbid < mid:
                    target_bid = overbid
                    break
                if bp < mid:
                    target_bid = bp
                    break
            if target_bid is not None:
                orders.append(Order("INTARIAN_PEPPER_ROOT", target_bid, buy_budget))

        return orders


    def _osmium(
        self,
        od: OrderDepth,
        pos: int,
        limit: int,
        td: dict,
    ) -> Tuple[List[Order], dict]:
        orders: List[Order] = []

        if od.buy_orders:
            bid_wall = min(od.buy_orders.keys())
            best_bid = max(od.buy_orders.keys())
        else:
            bid_wall = best_bid = None

        if od.sell_orders:
            ask_wall = max(od.sell_orders.keys())
            best_ask = min(od.sell_orders.keys())
        else:
            ask_wall = best_ask = None

        if bid_wall is not None and ask_wall is not None:
            wall_mid = (bid_wall + ask_wall) / 2
        elif best_bid is not None and best_ask is not None:
            wall_mid = (best_bid + best_ask) / 2
        elif best_bid is not None:
            wall_mid = float(best_bid)
        elif best_ask is not None:
            wall_mid = float(best_ask)
        else:
            return orders, td

        prev_fv = td.get("osm_fv", OSM_ANCHOR)
        fv = (1 - OSM_EMA_ALPHA) * prev_fv + OSM_EMA_ALPHA * wall_mid
        fv = max(OSM_ANCHOR - OSM_FV_CLAMP, min(OSM_ANCHOR + OSM_FV_CLAMP, fv))
        td["osm_fv"] = fv

        buy_budget  = limit - pos
        sell_budget = limit + pos

        # ── 1. AGGRESSIVE TAKING ────────────────────────────
        if od.sell_orders and buy_budget > 0:
            for ask in sorted(od.sell_orders.keys()):
                vol_avail = abs(od.sell_orders[ask])
                if ask <= fv - 1:
                    qty = min(vol_avail, buy_budget)
                    if qty > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", ask, qty))
                        buy_budget -= qty
                elif ask <= fv and pos < 0:
                    qty = min(vol_avail, abs(pos), buy_budget)
                    if qty > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", ask, qty))
                        buy_budget -= qty

        if od.buy_orders and sell_budget > 0:
            for bid in sorted(od.buy_orders.keys(), reverse=True):
                vol_avail = od.buy_orders[bid]
                if bid >= fv + 1:
                    qty = min(vol_avail, sell_budget)
                    if qty > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", bid, -qty))
                        sell_budget -= qty
                elif bid >= fv and pos > 0:
                    qty = min(vol_avail, pos, sell_budget)
                    if qty > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", bid, -qty))
                        sell_budget -= qty

        # ── 2. PASSIVE QUOTING WITH INVENTORY SKEW ─────────
        # NEW: skew shifts both quotes toward reducing inventory.
        # If pos = +80 (max long):  skew = -3  → bid/ask both go down 3
        #                            → less eager to buy, more eager to sell
        # If pos = -80 (max short): skew = +3  → bid/ask both go up 3
        #                            → more eager to buy, less eager to sell
        if bid_wall is not None and ask_wall is not None:
            bid_price = int(bid_wall + 1)
            ask_price = int(ask_wall - 1)

            for bp in sorted(od.buy_orders.keys(), reverse=True):
                overbid = bp + 1
                bv = od.buy_orders[bp]
                if bv > 1 and overbid < fv:
                    bid_price = max(bid_price, overbid)
                    break
                elif bp < fv:
                    bid_price = max(bid_price, bp)
                    break

            for sp in sorted(od.sell_orders.keys()):
                undercut = sp - 1
                sv = abs(od.sell_orders[sp])
                if sv > 1 and undercut > fv:
                    ask_price = min(ask_price, undercut)
                    break
                elif sp > fv:
                    ask_price = min(ask_price, sp)
                    break

            # Apply inventory skew: long position → push quotes down
            skew = -int((pos / limit) * OSM_SKEW_FACTOR)
            bid_price += skew
            ask_price += skew

            if buy_budget > 0:
                orders.append(Order("ASH_COATED_OSMIUM", bid_price, buy_budget))
            if sell_budget > 0:
                orders.append(Order("ASH_COATED_OSMIUM", ask_price, -sell_budget))

        return orders, td