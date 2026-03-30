from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

class Trader:
    POSITION_LIMITS = {"EMERALDS": 80, "TOMATOES": 80}
    EMERALD_FAIR = 10000
    SPREAD = 1  

    def run(self, state: TradingState):
        history = json.loads(state.traderData) if state.traderData else {}
        result = {}

        for product, order_depth in state.order_depths.items():
            if product not in self.POSITION_LIMITS:
                continue

            orders: List[Order] = []
            pos = state.position.get(product, 0)
            limit = self.POSITION_LIMITS[product]

            sells = order_depth.sell_orders 
            buys = order_depth.buy_orders  

            best_ask = min(sells) if sells else None
            best_bid = max(buys) if buys else None

            if product not in history:
                history[product] = {"mids": [], "ema": None}

            if best_ask and best_bid:
                mid = (best_ask + best_bid) / 2
                history[product]["mids"].append(mid)
                history[product]["mids"] = history[product]["mids"][-100:]

            mids = history[product]["mids"]

            #market making
            if product == "EMERALDS":
                fair = self.EMERALD_FAIR

                if best_ask is not None and best_ask < fair:
                    qty = min(-sells[best_ask], limit - pos)
                    if qty > 0:
                        orders.append(Order(product, best_ask, qty))
                        pos += qty

                if best_bid is not None and best_bid > fair:
                    qty = min(buys[best_bid], limit + pos)
                    if qty > 0:
                        orders.append(Order(product, best_bid, -qty))
                        pos -= qty

                skew = round(pos / limit * 2) 
                my_bid = fair - self.SPREAD - skew
                my_ask = fair + self.SPREAD - skew

                bid_qty = limit - pos
                ask_qty = limit + pos

                if bid_qty > 0:
                    orders.append(Order(product, my_bid, bid_qty))
                if ask_qty > 0:
                    orders.append(Order(product, my_ask, -ask_qty))


            elif product == "TOMATOES":
                if len(mids) < 10:
                    result[product] = orders
                    continue

                fast = sum(mids[-5:]) / 5
                slow = sum(mids[-20:]) / 20 if len(mids) >= 20 else sum(mids) / len(mids)
                trend = fast - slow  

                n = min(10, len(mids))
                xs = list(range(n))
                ys = mids[-n:]
                xmean = sum(xs) / n
                ymean = sum(ys) / n
                slope = sum((x - xmean) * (y - ymean) for x, y in zip(xs, ys)) / \
                        sum((x - xmean) ** 2 for x in xs)

                fair = fast  

                if slope > 0.5 and trend > 0:
                    if best_ask and best_ask < fair + 1:
                        qty = min(-sells[best_ask], limit - pos)
                        if qty > 0:
                            orders.append(Order(product, best_ask, qty))

                    if pos < limit:
                        orders.append(Order(product, int(fair) - 2, min(15, limit - pos)))

                elif slope < -0.5 and trend < 0:
                    if best_bid and best_bid > fair - 1:
                        qty = min(buys[best_bid], limit + pos)
                        if qty > 0:
                            orders.append(Order(product, best_bid, -qty))

                    if pos > -limit:
                        orders.append(Order(product, int(fair) + 2, -min(15, limit + pos)))

                else:
                    if best_ask and best_ask < fair - 1:
                        qty = min(-sells[best_ask], limit - pos, 25)
                        if qty > 0:
                            orders.append(Order(product, best_ask, qty))
                    if best_bid and best_bid > fair + 1:
                        qty = min(buys[best_bid], limit + pos, 25)
                        if qty > 0:
                            orders.append(Order(product, best_bid, -qty))

            result[product] = orders

        return result, 0, json.dumps(history)