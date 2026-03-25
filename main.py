
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
            
            current_position = state.position[product]

            past_average_price = state.own_trades[product].price / state.own_trades[product].quantity
            current_average_price = state.observations.conversionObservations[product].bidPrice / state.observations.conversionObservations[product].quantity

            acceptable_price =   # Participant should calculate this value



            
        
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
