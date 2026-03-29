from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import json

class Trader:

    position_limits={
        "EMERALDS":80,
        "TOMATOES":80.
    }
    emeralf_fair_price = 10000
    moving_avg_window = 20

    def bid(self):
        return 15
    
    def run(self, state: TradingState):
        history = json.loads(state.traderData) if state.traderData else {}
        result = {}
        
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        for product in state.order_depths:

            if product not in self.position_limits:
                continue

            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            current_position = state.position.get(product, 0)

            max_position = self.position_limits[product]

            if product not in history:
                history[product] = []

            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                history[product].append(best_ask)
            else:
                best_ask = None
                best_ask_amount = None
            
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
            else:
                best_bid = None
                best_bid_amount = None


            if product == "EMERALDS":
                fair_price = self.emerald_fair_price

            elif product == "TOMATOES":
                prices = history[product]
                if len(prices) >= self.moving_avg_window:
                    fair_price = sum(prices[-self.moving_avg_window:])/self.moving_avg_window
                elif len(prices)>0:
                    fair_price = sum(prices)/len(prices)
                else:
                    result[product] = orders
                    continue
          

            # BUY
            if best_ask is not None:
                if int(best_ask) < fair_price:
                    buyable = max_position - current_position
                    buy_qty = min(-best_ask_amount, buyable)

                    if buy_qty > 0:
                        print("BUY", str(buy_qty) + "x", best_ask)
                        orders.append(Order(product, best_ask, buy_qty))

            # SELL
            if best_bid is not None:
                if int(best_bid) > fair_price:
                    sellable = max_position + current_position
                    sell_qty = min(best_bid_amount, sellable)

                    if sell_qty > 0:
                        print("SELL", str(sell_qty) + "x", best_bid)
                        orders.append(Order(product, best_bid, -sell_qty))

            result[product] = orders

            if orders:
                print(f"{product}: sending {orders}")
            result[product] = orders


    
        traderData = json.dumps(history)
        conversions = 0
        return result, conversions, traderData