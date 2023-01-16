import numpy as np
import logging

logger = logging.getLogger(__name__)

class Broker:

    def __init__(self, cash_idx, price_idxs, depot_idxs, buy_price_idxs, stock_dim, transaction_cost):
        self.cash_idx = cash_idx
        self.price_idxs = price_idxs
        self.depot_idxs = depot_idxs
        self.buy_price_idxs = buy_price_idxs
        self.stock_dim = stock_dim
        self.transaction_cost = transaction_cost

        if self.transaction_cost == 0:
            logger.warning("Currently assuming NO transaction cost!!")


    def calculate_total_asset_value(self, state):

        cash = state[self.cash_idx]
        prices = np.array(state[self.price_idxs])
        depot = np.array(state[self.depot_idxs])
        depot_value = sum(prices * depot)

        return cash + depot_value


    def _get_max_amnt_pct_thresh(self, state, price, current_shares, cap_pct):

        max_value_per_asset = self.calculate_total_asset_value(state) * cap_pct
        current_value = price * current_shares
        if current_value != 0:
            pause = 1

        remaining_value = max_value_per_asset - current_value

        if remaining_value <= 0:
            return 0

        max_share = remaining_value / price
        max_share = int(max_share)

        return max_share


    def _update_buy_prices(self, buy_price, buy_num_shares, owned_shares, price):


        both_prices = [buy_price, price]
        both_amounts = [owned_shares, buy_num_shares]

        weighted_avg = np.average(both_prices, weights=both_amounts)

        return weighted_avg


    def buy_stock(self, index, amount, state, cap_pct=None):
        """
    
        Parameters
        ----------
        index - The index of the stock to buy in the state
        action - how many stocks to buy
    
        Returns - the amount of stocks to buy
        -------
    
        """

        price = np.array(state)[self.price_idxs][index]
        owned_shares = np.array(state)[self.depot_idxs][index]
        buy_price = np.array(state)[self.buy_price_idxs][index]

        # check if the stock is able to buy - buy only if the price is > 0 (no missing data in this particular date)
        if state[index + 2 * self.stock_dim + 1] != True:
            # Check if enough cash is available - Dicide vaialbel cash by stock_price
            possible_shares = int(state[self.cash_idx] // (price * (1 + self.transaction_cost)))
            # Limit stock amouint to pct_ap
            amnt_pct_cap = 1000 if cap_pct is None else self._get_max_amnt_pct_thresh(state, price, owned_shares, cap_pct)
            # Define how many stocks to buy
            buy_num_shares = min([possible_shares, amount, amnt_pct_cap])
            # Update the buy prices based on wieghted mean if shares are supposed to be bought
            if buy_num_shares > 0:
                buy_price = self._update_buy_prices(buy_price, buy_num_shares, owned_shares, price)
            # Calculate the price of the transaction
            buy_amount = price * buy_num_shares * (1 + self.transaction_cost)
            # Update the cash and the depot
            # state[self.cash_idx] -= buy_amount
            owned_shares += buy_num_shares
            # Calculate cost of this trade
            transaction_cost = price * buy_num_shares * self.transaction_cost

            logger.debug("Tried to buy {} shares. Will buy {} at {} for {}. Depot {} -> {}. TranactionCost: {}".format(
                amount, buy_num_shares, price, buy_amount, np.array(state)[self.depot_idxs][index], owned_shares,
                transaction_cost))

        else:
            buy_num_shares = 0
            transaction_cost = 0
            buy_amount = 0

            logger.debug("Tried to buy shares but: state[index + 2 * self.stock_dim + 1] == True")

        depot = np.array(state)[self.depot_idxs]
        depot[index] = owned_shares
        buy_prices = np.array(state)[self.buy_price_idxs]
        buy_prices[index] = buy_price

        return buy_num_shares, depot, buy_prices, buy_amount, transaction_cost

    def sell_stock(self, index, amount, state):

        price = np.array(state)[self.price_idxs][index]
        owned_shares = np.array(state)[self.depot_idxs][index]
        buy_price = np.array(state)[self.buy_price_idxs][index]

        # check if the stock is able to sell, for simlicity we just add it in techical index
        if (state[index + 2 * self.stock_dim + 1] != True):
            # if state[index + 1] > 0: # if we use price<0 to denote a stock is unable to trade in that day, the total asset calculation may be wrong for the price is unreasonable

            # Sell only if current asset is > 0
            if owned_shares > 0:
                sell_num_shares = min(abs(amount), owned_shares)
                # Calculate the return of this sale
                sell_amount = price * sell_num_shares
                roi = (price - buy_price) * sell_num_shares
                taxes = np.max([roi * 0.25, 0])  # only taxes if positive trade
                # update balance
                # state[self.cash_idx] += sell_amount
                # Remove number of share from depot
                owned_shares -= sell_num_shares
                # Keep track of trade cost and trades
                transaction_cost = taxes + 1 # 1 Euro transaction costs
                # apply cost to sell_amount
                sell_amount -= transaction_cost

                logger.debug(
                    "Trying to sell {} shares. Will sell {} at {} for {}. Bought for {}. Depot {} -> {}. TranactionCost: {}".format(
                        amount, sell_num_shares, price, sell_amount, buy_price, np.array(state)[self.depot_idxs][index],
                        owned_shares,transaction_cost))

            else:
                sell_num_shares = 0
                transaction_cost = 0
                sell_amount = 0
                logger.debug("Trying to sell {} shares. But none are in depot. Skipping".format(amount))
        else:
            sell_num_shares = 0
            transaction_cost = 0
            sell_amount = 0


        depot = np.array(state)[self.depot_idxs]
        depot[index] = owned_shares

        return -sell_num_shares, depot, sell_amount, transaction_cost



def sell_all(self, state):

    prices = state[self.price_idxs]
    depot = state[self.depot_idxs]

    if self.turbulence_threshold is not None:
        # Turbulence is  takingplace
        if self.turbulence >= self.turbulence_threshold:
            if state[index + 1] > 0:
                # Sell only if the price is > 0 (no missing data in this particular date)
                # if turbulence goes over threshold, just clear out all positions
                if depot[index] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = depot[index]
                    sell_amount = (
                            state[index + 1]
                            * sell_num_shares
                            * (1 - self.transaction_cost)
                    )
                    # update balance
                    state[0] += sell_amount
                    depot[index] = 0
                    self.cost += (
                            state[index + 1]
                            * sell_num_shares
                            * self.transaction_cost
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0
        # Turbulence threshold set but not exceeded
        else:
            sell_num_shares = _do_sell_normal()



# def _buy_stock(index, action):
#     """
#
#     Parameters
#     ----------
#     index - The index of the stock to buy in the state
#     action - how many stocks to buy
#
#     Returns - the amount of stocks to buy
#     -------
#
#     """
#     def _do_buy():
#         # check if the stock is able to buy - buy only if the price is > 0 (no missing data in this particular date)
#         if state[index + 2 * self.stock_dim + 1] != True:
#
#             # Check if enough cash is available - Dicide vaialbel cash by stock_price
#             available_amount = state[0] // (state[index + 1] * (1 + self.transaction_cost))
#             # Define how many stocks to buy
#             buy_num_shares = min(available_amount, action)
#             # Calculate the price of the transaction
#             buy_amount = (state[index + 1] * buy_num_shares * (1 + self.transaction_cost))
#             # Update the cash
#             state[0] -= buy_amount
#             # Update the state
#             state[index + self.stock_dim + 1] += buy_num_shares
#
#             self.cost += (
#                     state[index + 1] * buy_num_shares * self.transaction_cost
#             )
#             self.trades += 1
#         else:
#             buy_num_shares = 0
#
#         return buy_num_shares
#
#     # perform buy action based on the sign of the action
#     if self.turbulence_threshold is None:
#         buy_num_shares = _do_buy()
#     else:
#         if self.turbulence < self.turbulence_threshold:
#             buy_num_shares = _do_buy()
#         else:
#             buy_num_shares = 0
#             pass
#
#     return buy_num_shares
#
#
#
# def _sell_stock(index, action):
#     def _do_sell_normal():
#         if (
#             state[index + 2 * self.stock_dim + 1] != True
#         ):  # check if the stock is able to sell, for simlicity we just add it in techical index
#             # if state[index + 1] > 0: # if we use price<0 to denote a stock is unable to trade in that day, the total asset calculation may be wrong for the price is unreasonable
#             # Sell only if the price is > 0 (no missing data in this particular date)
#             # perform sell action based on the sign of the action
#             if state[index + self.stock_dim + 1] > 0:
#                 # Sell only if current asset is > 0
#                 sell_num_shares = min(
#                     abs(action), state[index + self.stock_dim + 1]
#                 )
#                 sell_amount = (
#                     state[index + 1]
#                     * sell_num_shares
#                     * (1 - self.transaction_cost)
#                 )
#                 # update balance
#                 state[0] += sell_amount
#
#                 state[index + self.stock_dim + 1] -= sell_num_shares
#                 self.cost += (
#                     state[index + 1]
#                     * sell_num_shares
#                     * self.transaction_cost
#                 )
#                 self.trades += 1
#             else:
#                 sell_num_shares = 0
#         else:
#             sell_num_shares = 0
#
#         return sell_num_shares
#
#     # perform sell action based on the sign of the action
#     if self.turbulence_threshold is not None:
#         if self.turbulence >= self.turbulence_threshold:
#             if state[index + 1] > 0:
#                 # Sell only if the price is > 0 (no missing data in this particular date)
#                 # if turbulence goes over threshold, just clear out all positions
#                 if state[index + self.stock_dim + 1] > 0:
#                     # Sell only if current asset is > 0
#                     sell_num_shares = state[index + self.stock_dim + 1]
#                     sell_amount = (
#                         state[index + 1]
#                         * sell_num_shares
#                         * (1 - self.transaction_cost)
#                     )
#                     # update balance
#                     state[0] += sell_amount
#                     state[index + self.stock_dim + 1] = 0
#                     self.cost += (
#                         state[index + 1]
#                         * sell_num_shares
#                         * self.transaction_cost
#                     )
#                     self.trades += 1
#                 else:
#                     sell_num_shares = 0
#             else:
#                 sell_num_shares = 0
#         else:
#             sell_num_shares = _do_sell_normal()
#     else:
#         sell_num_shares = _do_sell_normal()
#
#     return sell_num_shares
