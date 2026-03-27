from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import json

class Trader:

    def bid(self):
        return 15
    
    def run(self, state: TradingState):
        history = json.loads(state.traderData) if state.traderData else {}
        result = {}
        
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            current_position = state.position.get(product, 0)

            if product == "EMERALDS":
                fair_price = 10000
                max_position = 80
            elif product == "TOMATOES":
                fair_price = 5000
                max_position = 80
            else:
                result[product] = orders
                continue

            print("Buy Order depth : " + str(len(order_depth.buy_orders)) +
                  ", Sell order depth : " + str(len(order_depth.sell_orders)))

            # BUY
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < fair_price:
                    buyable = max_position - current_position
                    buy_qty = min(-best_ask_amount, buyable)

                    if buy_qty > 0:
                        print("BUY", str(buy_qty) + "x", best_ask)
                        orders.append(Order(product, best_ask, buy_qty))

            # SELL
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > fair_price:
                    sellable = max_position + current_position
                    sell_qty = min(best_bid_amount, sellable)

                    if sell_qty > 0:
                        print("SELL", str(sell_qty) + "x", best_bid)
                        orders.append(Order(product, best_bid, -sell_qty))

            result[product] = orders

            if product not in history:
                history[product] = []

            if len(order_depth.sell_orders) != 0:
                history[product].append(best_ask)
    
        traderData = json.dumps(history)
        conversions = 0
        return result, conversions, traderData