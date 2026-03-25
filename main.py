
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string



class Trader:

    def bid(self):
        return 15
    
    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""

        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            if product not in state.own_trades.keys():
                #initialize own trade history, can use to calc moving average
#idk how to do this bruh


            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            
            current_position = state.position.get(product, 0)


            #probs not good method to calc average price
            current_mid_price = (best_bid + best_ask) / 2
                   #fix these conditions later
            
            if product not in history:
                history[product] = []

            history[product].append(current_price)

            # keep only last 10 prices
            history[product] = history[product][-10:]

            # moving average
            acceptable_price = sum(history[product]) / len(history[product])



            
                   
            if current_position > 0 and current_mid_price > #something: # expecting that we are buying low, selling high
                acceptable_price = 
        
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:

                    orders.append(Order(product, best_ask, -best_ask_amount))
    
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list[tuple[int, int]](order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
    
                    orders.append(Order(product, best_bid, -best_bid_amount))
            
            result[product] = orders
    
        traderData = ""  # No state needed - we check position directly
        conversions = 0
        return result, conversions, traderData
