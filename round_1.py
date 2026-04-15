from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

"""
Strategy:
- INTARIAN_PEPPER_ROOT: Market make around fair value.
  Fair value is very stable within a day (~10000, 11000, 12000 on days -2/-1/0).
  We estimate fair value from the order book mid-price and undercut the spread.

- ASH_COATED_OSMIUM: Mean reversion around ~10000.
  Price oscillates within ~±20 of 10000. Buy when below fair value, sell above.
  Also market make around fair value to capture spread.
"""

POSITION_LIMITS = {
    "INTARIAN_PEPPER_ROOT": 80,
    "ASH_COATED_OSMIUM": 80,
}

# ASH fair value is stable at ~10000
ASH_FAIR_VALUE = 10000

class Trader:

    def run(self, state: TradingState):
        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            position = state.position.get(product, 0)
            limit = POSITION_LIMITS[product]

            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

            if product == "INTARIAN_PEPPER_ROOT":
                orders = self.trade_pepper(order_depth, position, limit, best_bid, best_ask)

            elif product == "ASH_COATED_OSMIUM":
                orders = self.trade_osmium(order_depth, position, limit, best_bid, best_ask)

            result[product] = orders

        return result, 0, ""

    def trade_pepper(
        self,
        order_depth: OrderDepth,
        position: int,
        limit: int,
        best_bid,
        best_ask,
    ) -> List[Order]:
        """
        INTARIAN_PEPPER_ROOT is steady within a day.
        Estimate fair value from mid price and market make 1 tick inside the spread.
        Also aggressively hit any mispricings (bids above or asks below fair value).
        """
        orders = []

        if best_bid is None and best_ask is None:
            return orders

        # Estimate fair value from order book
        if best_bid and best_ask:
            fair_value = (best_bid + best_ask) / 2
        elif best_bid:
            fair_value = best_bid + 7
        else:
            fair_value = best_ask - 7

        # --- Take mispriced orders ---
        # Buy anything offered below fair value
        if best_ask and best_ask < fair_value:
            buy_qty = min(
                abs(order_depth.sell_orders[best_ask]),
                limit - position,
            )
            if buy_qty > 0:
                orders.append(Order(product="INTARIAN_PEPPER_ROOT", price=best_ask, quantity=buy_qty))
                position += buy_qty

        # Sell anything bid above fair value
        if best_bid and best_bid > fair_value:
            sell_qty = min(
                order_depth.buy_orders[best_bid],
                limit + position,
            )
            if sell_qty > 0:
                orders.append(Order(product="INTARIAN_PEPPER_ROOT", price=best_bid, quantity=-sell_qty))
                position -= sell_qty

        # --- Market make 1 tick inside spread ---
        mm_bid = int(fair_value) - 1
        mm_ask = int(fair_value) + 1

        buy_capacity = limit - position
        sell_capacity = limit + position

        if buy_capacity > 0:
            orders.append(Order(product="INTARIAN_PEPPER_ROOT", price=mm_bid, quantity=buy_capacity))

        if sell_capacity > 0:
            orders.append(Order(product="INTARIAN_PEPPER_ROOT", price=mm_ask, quantity=-sell_capacity))

        return orders

    def trade_osmium(
        self,
        order_depth: OrderDepth,
        position: int,
        limit: int,
        best_bid,
        best_ask,
    ) -> List[Order]:
        """
        ASH_COATED_OSMIUM oscillates around 10000.
        Mean-revert: buy aggressively when price dips, sell when it rises.
        Also market make around fair value.
        """
        orders = []
        fair_value = ASH_FAIR_VALUE

        # --- Take mispriced orders (mean reversion) ---
        # Buy if ask is below fair value (price is depressed)
        if best_ask and best_ask < fair_value:
            buy_qty = min(
                abs(order_depth.sell_orders[best_ask]),
                limit - position,
            )
            if buy_qty > 0:
                orders.append(Order(product="ASH_COATED_OSMIUM", price=best_ask, quantity=buy_qty))
                position += buy_qty

        # Sell if bid is above fair value (price is elevated)
        if best_bid and best_bid > fair_value:
            sell_qty = min(
                order_depth.buy_orders[best_bid],
                limit + position,
            )
            if sell_qty > 0:
                orders.append(Order(product="ASH_COATED_OSMIUM", price=best_bid, quantity=-sell_qty))
                position -= sell_qty

        # --- Market make 2 ticks inside spread around fair value ---
        # Use tighter spread than the existing ~16-18 tick market spread
        mm_bid = fair_value - 2
        mm_ask = fair_value + 2

        buy_capacity = limit - position
        sell_capacity = limit + position

        if buy_capacity > 0:
            orders.append(Order(product="ASH_COATED_OSMIUM", price=mm_bid, quantity=buy_capacity))

        if sell_capacity > 0:
            orders.append(Order(product="ASH_COATED_OSMIUM", price=mm_ask, quantity=-sell_capacity))

        return orders