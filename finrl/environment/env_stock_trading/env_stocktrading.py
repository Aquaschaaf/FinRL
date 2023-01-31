from __future__ import annotations

import copy
from typing import List

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from finrl.plot import plot_actions, plot_states
from scipy.stats import entropy
from finrl.environment.env_stock_trading.Broker import Broker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import logging
import math

logger = logging.getLogger(__name__)
matplotlib.use("Agg")

# from stable_baselines3.common.logger import Logger, KVWriter, CSVOutputFormat


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
        idle_threshold=None
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount  # get the initial cash
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        # initalize state
        self.state = self._initiate_state()
        self.idle_threshold = idle_threshold
        self.performance_all_trades = 0
        self.all_buy_price_low_rewards = 0
        self.last_sale = 0

        self.no_trades_counter = 0
        self.cash_idx = 0
        self.all_buy_Low_sellHigh_rewards = 0
        self.price_idxs = np.array(range(1, self.stock_dim + 1))
        self.depot_idxs = np.array(range(self.stock_dim + 1, self.stock_dim * 2 + 1))
        self.buy_price_idxs = np.array(range(self.stock_dim * 2 + 1, self.stock_dim * 3 + 1))
        self.tech_indicator_start_idx = self.stock_dim * 3 + 1
        self.broker = Broker(cash_idx=self.cash_idx,
                             price_idxs=self.price_idxs,
                             buy_price_idxs=self.buy_price_idxs,
                             depot_idxs=self.depot_idxs,
                             stock_dim=self.stock_dim,
                             transaction_cost=0.03)  # 0.03

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
        ]  # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
        self.rewards_memory = []
        self.actions_memory = []
        self.orig_actions_memory = []
        self.state_memory = (
            []
        )  # we need sometimes to preserve the state in the middle of trading process
        self.date_memory = [self._get_date()]
        #         self.logger = Logger('results',[CSVOutputFormat])
        # self.reset()
        self._seed()


    def _update_depot(self, actions):

        argsort_actions = np.argsort(actions)
        sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

        # Don't buy anything if turbulenced
        if self.turbulence_threshold is not None and self.turbulence > self.turbulence_threshold:
            # sell all stocks
            logger.warning("Missing implementeation of selling all stock in turbulent times")
        else:

            for index in sell_index:

                actions[index], depot, sale_return, cost = self.broker.sell_stock(index, actions[index], self.state)
                self.cost += cost
                self.state[self.cash_idx] += sale_return
                self.state[self.depot_idxs] = depot
                if self.state[self.depot_idxs][index] == 0:
                    self.state[self.buy_price_idxs][index] = 0
                if abs(actions[index]) > 0:
                    self.trades += 1

                self.last_sale = self.day

                self.sold_shares = True

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                actions[index], depot, buy_prices, buy_cost, cost = self.broker.buy_stock(index, actions[index], self.state)
                self.state[self.cash_idx] -= buy_cost
                self.state[self.depot_idxs] = depot
                self.state[self.buy_price_idxs] = buy_prices
                self.cost += cost
                # if abs(actions[index]) > 0:
                #     self.trades += 1

        return actions

    def calculate_performance(self, old_val, new_val):
        return (new_val - old_val) / abs(old_val)



    def calculate_total_asset_value(self, state):

        cash = state[self.cash_idx]
        depot_value = self.calculate_total_depot_value(state)

        return cash + depot_value

    def calculate_total_depot_value(self, state):

        prices = np.array(state[self.price_idxs])
        depot = np.array(state[self.depot_idxs])
        return sum(prices * depot)

    def calculate_depot_performance(self, state):

        prices = np.array(state[self.price_idxs])
        buy_prices = np.array(state[self.buy_price_idxs])
        depot = np.array(state[self.depot_idxs])

        price_diffs = prices - buy_prices
        return sum(price_diffs * depot)

    def calculate_depot_performance_pct(self):

        depot = np.array(self.state[self.depot_idxs])
        prices = np.array(self.state[self.price_idxs])
        buy_prices = np.array(self.state[self.buy_price_idxs])

        nonzero = np.nonzero(depot)
        prices = prices[nonzero]
        buy_prices = buy_prices[nonzero]

        price_diffs = (prices - buy_prices) / buy_prices * 100
        return sum(price_diffs)


    def plot_state_memory(self):
        all_cash = [sm[self.cash_idx] for sm in self.state_memory]
        all_prices = [sm[self.price_idxs][0] for sm in self.state_memory]
        all_depots = [sm[self.depot_idxs][0] for sm in self.state_memory]
        buy_prices = [sm[self.buy_price_idxs][0] for sm in self.state_memory]
        all_total_value = [self.calculate_total_asset_value(sm) for sm in self.state_memory]
        all_features = [list(sm[self.buy_price_idxs[-1] + 1:]) for sm in self.state_memory]
        f1, f2, f3, f4, f5, f6, f7, f8, f9 = map(list, zip(*all_features))

        fig, axs = plt.subplots(4,1)
        axs[0].plot(all_cash)
        axs[1].plot(all_prices)
        axs[1].plot(buy_prices)
        axs[2].plot(all_depots)
        axs[3].plot(all_total_value)

        fig, ax = plt.subplots(1,1)
        ax.plot(f1)
        ax.plot(f2)
        ax.plot(f3)
        ax.plot(f4)
        ax.plot(f5)
        ax.plot(f6)
        ax.plot(f7)
        ax.plot(f8)
        ax.plot(f9)
        plt.show()


    def calc_portfolio_weight_entropy(self):

        prices = np.array(self.state[self.price_idxs])
        depot = np.array(self.state[self.depot_idxs])
        all_investments = sum(prices * depot)
        weights_per_stock = []

        if all_investments == 0.0:
            return 0.0

        for price, amount in zip(prices, depot):

            investment_per_stock = price * amount
            weights_per_stock.append(investment_per_stock / all_investments)

        weight_entropy = entropy(weights_per_stock)
        if math.isnan(weight_entropy):
            return 0.0

        return weight_entropy

    def conv_state2string(self, state):

        out_string = "Cash: {}, #Shares: {}, Prices: {}, BuyPrices: {}".format(
            state[self.cash_idx],
            state[self.depot_idxs],
            state[self.price_idxs],
            state[self.buy_price_idxs])
        return out_string


    def _make_plot(self):

        img_states = plot_states(self.save_state_memory())
        # plt.imshow(img_states, interpolation='nearest')
        # plt.savefig(f"results/states_episode{self.episode}.png")

        img_actions = plot_actions(self.df, self.save_action_memory())
        # plt.imshow(img, interpolation='nearest')
        # plt.savefig(f"results/actions_episode{self.episode}.png")

        return img_states, img_actions


    def step(self, actions):

        logger.debug("Processing step {}".format(self.day))

        self.terminal = self.day >= len(self.df.index.unique()) - 1

        all_assets_value = self.calculate_total_asset_value(self.state)
        if all_assets_value < 1000:
            print("IM BROKE!")
            self.terminal = True

        # idle_end = False
        # # If its not the official end, check if its a no trade end
        # if not self.terminal and self.idle_threshold is not None:
        #     self.terminal = self.no_trades_counter > self.idle_threshold
        #     idle_end = True
        #     if self.terminal:
        #         print("ENDING EPISODE DUE TO IDELING AFTER {} DAYS".format(self.day))

        # If done
        if self.terminal:
            # print(f"Episode: {self.episode}")

            if self.make_plots and self.episode % 15 == 0:
                img_states, img_actions = self._make_plot()
            else:
                img_states, img_actions = None, None
            end_total_asset = self.calculate_total_asset_value(self.state)

            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (end_total_asset - self.asset_memory[0])  # initial_amount is only cash part of our initial asset
            final_perf_pct = (end_total_asset - self.asset_memory[0]) / abs(self.asset_memory[0]) * 100

            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )
            else:
                sharpe = -3

            # Is this okay?
            if not self.rewards_memory:
                self.rewards_memory = {self.day: 0}

            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "results/actions_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                df_total_value.to_csv(
                    "results/account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    "results/account_rewards_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    "results/account_value_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.close()

            # Add outputs to logger interface
            # logger.record("environment/portfolio_value", end_total_asset)
            # logger.record("environment/total_reward", tot_reward)
            # logger.record("environment/total_reward_pct", (tot_reward / (end_total_asset - tot_reward)) * 100)
            # logger.record("environment/total_cost", self.cost)
            # logger.record("environment/total_trades", self.trades)

            # if not idle_end:
            # Add average Perfromance of all trades at end of episode
            avg_trade_perf = self.performance_all_trades / self.trades if self.trades != 0 else 0
            # print("Avergae trade performance at end of epidoe: {0:.2f} (total: {1:.2f} / {2:} trades)".format(
            #     avg_trade_perf, self.performance_all_trades, self.trades))
            # self.reward += avg_trade_perf

            self.reward = 0
            self.reward += avg_trade_perf * 5
            self.reward += final_perf_pct / 10
            # self.reward = tot_reward


            info = {"terminal/reward": self.reward,
                    "terminal/sharpe_ratio": sharpe,
                    "terminal/average_trade_perf_reward": avg_trade_perf,
                    "terminal/depot_performance": tot_reward,
                    "terminal/episode_length": self.day,
                    "terminal/number_of_trades": self.trades,
                    "terminal/avg_buyLow_sellHigh_rewards": self.all_buy_Low_sellHigh_rewards / self.trades if self.trades !=0 else 0,
                    "terminal/avg_train_rewards": np.mean(self.rewards_memory),
                    "terminal/avg_buy_price_low_rewards": self.all_buy_price_low_rewards / self.day,
                    "terminal/final_perf_pct": final_perf_pct,
                    "histograms/actions": [a[0] for a in self.orig_actions_memory],
                    "histograms/rewards": self.rewards_memory
                    }
            if img_states is not None:
                info["images/states"] = img_states
            if img_actions is not None:
                info["images/actions"] = img_actions

            # self.plot_state_memory()

            logger.debug("End of episode\n")

            return self.state, self.reward, self.terminal, info

        else:

            if abs(actions[0]) < 0.1:
                actions[0] = 0

            orig_action = copy.deepcopy(actions)
            self.orig_actions_memory.append(orig_action)
            # action = Amount of stocks to buy/sell
            # actions = actions * self.hmax  # actions initially is scaled between 0 to 1
            # actions = actions.astype(int)  # convert into integer because we can't by fraction of shares

            # actions[actions>0] = 1000
            # actions[actions<0] = -1000

            # state = [Cash, Value of Stocks, Num of stocks in Depot, TechnIndicator]
            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)

            begin_total_asset = self.calculate_total_asset_value(self.state)
            begin_depot_value = self.calculate_total_depot_value(self.state)
            begin_depot_performance = self.calculate_depot_performance(self.state)

            depot_before = self.state[self.depot_idxs]
            buy_prices_before = self.state[self.buy_price_idxs]
            cash_before = self.state[self.cash_idx]
            logger.debug("State before applying actions\n{}\nCurrent asset value: {}".format(self.conv_state2string(self.state), begin_total_asset))

            actions = self._update_depot(actions)

            self.actions_memory.append(actions)
            depot_after = self.state[self.depot_idxs]
            buy_prices_after = self.state[self.buy_price_idxs]
            cash_after = self.state[self.cash_idx]

            # Actual begin is here? After buying/selling according to action. Then uzpdate is next days data
            begin_total_asset = self.calculate_total_asset_value(self.state)
            begin_depot_value = self.calculate_total_depot_value(self.state)
            begin_depot_performance = self.calculate_depot_performance(self.state)
            logger.debug("State after applying actions\n{}\nCurrent asset value: {}".format(self.conv_state2string(self.state), begin_total_asset))

            buyLow_sellHigh_reward = 0
            buying_penalty = 0
            sell_index = actions < 0
            buy_index = actions > 0

            for s_i in sell_index:
                if s_i:
                    buyLow_sellHigh_reward += self.data.normalized_close.values[s_i]

            for b_i in buy_index:
                if b_i:
                    # buyLow_reward = 1 if self.data.normalized_close < 0.8 else -5
                    # buyLow_sellHigh_reward += buyLow_reward
                    buyLow_sellHigh_reward += 1 - self.data.normalized_close.values[b_i]
                    buying_penalty = 1

            self.all_buy_Low_sellHigh_rewards += buyLow_sellHigh_reward


            if all(v == 0 for v in self.state[self.depot_idxs]):
                depot_empty = True
            else:
                depot_empty = False

            non_zeros = actions[actions!=0]
            if len(non_zeros)==0:
                self.no_trades_counter += 1
            else:
                self.no_trades_counter = 0

            # Rate sell tranactions -> Sell_price - Buy_price * Amount
            sell_index = np.where(actions < 0)[0]
            trade_reward = 0
            for i in sell_index:
                sell_price = self.state[self.price_idxs][i]
                buy_price = buy_prices_before[i]
                # amount = abs(actions[i])
                # diff = sell_price - buy_price
                # perf = diff * amount

                # Trade perfromance relative to total asset
                # value_diff = sell_price * abs(actions[i]) - buy_price *  abs(actions[i])
                # value_diff_pct = value_diff / begin_total_asset * 100
                # trade_reward += value_diff_pct * 10
                # self.performance_all_trades += value_diff_pct

                # Trade perofrmance relative to buy_price
                perf_pct = self.calculate_performance(buy_price, sell_price) * 100
                trade_reward += perf_pct * 5
                self.performance_all_trades += perf_pct


                # # Only positive trades
                # perf = np.max([0, perf])



            # trade_reward *= 10
            # trade_reward = np.max([trade_reward, 0])
            # maybe the cehck if trade reaward > 100 or so. Set to 0 if not -> Incetivise profitable trades

            # state: s -> s+1
            self.day += 1
            self.data = self.df.loc[self.day, :]
            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state = self._update_state()

            end_total_asset = self.calculate_total_asset_value(self.state)
            end_depot_value = self.calculate_total_depot_value(self.state)
            end_depot_performance = self.calculate_depot_performance(self.state)
            logger.debug("State after applying updating date:\n{}\nCurrent asset value: {}".format(self.conv_state2string(self.state), end_total_asset))

            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())

            weight_entropy = self.calc_portfolio_weight_entropy()
            diversification_reward = -0.5 + weight_entropy * 100
            depot_perf_reward = self.calculate_depot_performance_pct() #* 0.1
            # # ToDo Problem: Negative Performance am Beginn - dann kommt sell Befehl und Depot ist leer -> Performance ist 0 -> Ergibt einen positiven Reward. Inzentiviert verkaufen bei negativer Perofrmance? Gelöst durch das depot empty?
            # if begin_depot_performance != 0:
            #     perf_devel_pct = (end_depot_performance - begin_depot_performance) / abs(begin_depot_performance) * 100
            # else:
            #     perf_devel_pct = 0
            # depot_perf_reward = perf_devel_pct if not depot_empty else 0
            # depot_perf_reward = ((end_depot_performance - begin_depot_performance)/begin_depot_performance) * 100
            # depot_perf_reward = end_depot_value - begin_depot_value
            # depot_perf_reward = end_total_asset - begin_total_asset

            no_trade_reward = 0 if self.no_trades_counter < 40 else -10
            # no_trade_reward = 0

            buy_price_low_reward = np.sum(self.state[self.price_idxs] - self.state[self.buy_price_idxs])
            self.all_buy_price_low_rewards += buy_price_low_reward

            "https://ai.stackexchange.com/questions/10082/suitable-reward-function-for-trading-buy-and-sell-orders"
            self.reward = 0
            self.reward += depot_perf_reward
            # self.reward += diversification_reward
            self.reward += trade_reward
            # self.reward += buyLow_sellHigh_reward
            # self.reward += no_trade_reward
            # self.reward += buy_price_low_reward
            self.reward += buying_penalty
            # self.reward += fake_reward

            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling

            self.state_memory.append(self.state)  # add current state in state_recorder for each step

            info = {"reward/diversification_reward": diversification_reward,
                    "reward/depot_performance_reward": depot_perf_reward,
                    "reward/trade_reward": trade_reward,
                    "reward/buyLow_sellHigh_reward": buyLow_sellHigh_reward,
                    "reward/total_reward": self.reward,
                    "train/avg_perfroamce_all_trades": self.performance_all_trades/self.trades if self.trades !=0 else 0,
                    "train/num_episode": self.episode,
                    "train/no_trade_reward": no_trade_reward,
                    "train/actions": orig_action[0]

                    }

            logger.debug("End of step\n")

        return self.state, self.reward, self.terminal, info

    def reset(self):
        logger.debug("Resetting StockTradingEnv")
        # initiate state
        self.state = self._initiate_state()

        if self.initial:
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1 : 1 + self.stock_dim])
                )
            ]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]
        self.state[self.buy_price_idxs] = 0

        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.performance_all_trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.orig_actions_memory = []
        self.state_memory = []
        self.date_memory = [self._get_date()]
        self.no_trades_counter = 0
        self.all_buy_Low_sellHigh_rewards = 0
        self.all_buy_price_low_rewards = 0
        self.last_sale = 0

        self.episode += 1

        return self.state

    def render(self, mode="human", close=False):
        # plot_actions(self.df, self.actions_memory)
        return self.state

    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:

                sum_tech = sum( (self.data[tech].values.tolist() for tech in self.tech_indicator_list ), [] , )
                # for multiple stock
                state = (
                    [self.initial_amount]                   # intital cash
                    + self.data.close.values.tolist()       # close values
                    + self.num_stock_shares                 # Number of share per stock
                    + [0] * self.stock_dim                  # Buy prices
                    + sum_tech                              # Not sure what this is
                )  # append initial stocks_share to initial state, instead of all zero
            else:
                # for single stock
                state = (
                    [self.initial_amount]                   # intital cash
                    + [self.data.close]                     # close values
                    + [0] * self.stock_dim                  # Number of share per stock
                    + [0] * self.stock_dim                  # Buy prices
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])  # Not sure what this is
                )
        else:

            logger.error("USING NON INITIAL STATE SPACE INITITALIZATION: THIS WAS UPDATED AFTER ADDING BUY PRICE")
            exit()
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.previous_state[0]]
                    + self.data.close.values.tolist()
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + [0] + self.stock_dim  # buy prices
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )
            else:
                # for single stock
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        return np.array(state)

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = (
                [self.state[0]]
                + self.data.close.values.tolist()
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + list(self.state[self.buy_price_idxs])
                + sum(
                    (
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ),
                    [],
                )
            )

        else:
            # for single stock
            state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + list(self.state[self.buy_price_idxs])
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
            )

        return np.array(state)

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    # add save_state_memory to preserve state in the trading process
    def save_state_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]
            state_list = self.state_memory

            stocks = self.data.tic.values
            columns = ["col_{}".format(i) for i in range(len(self.state))]
            columns[self.cash_idx] = "cash"
            for idx, i in enumerate(self.depot_idxs):
                columns[i] = "Shares_{}".format(stocks[idx])
            for idx, i in enumerate(self.price_idxs):
                columns[i] = "Price_{}".format(stocks[idx])
            for idx, i in enumerate(self.buy_price_idxs):
                columns[i] = "BuyPrice_{}".format(stocks[idx])

            df_states = pd.DataFrame(
                state_list,
                columns=columns,
            )
            df_states["TotalValue"] = df_states.apply(lambda row: self.calculate_total_asset_value(row), axis=1)
            df_states.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]
            state_list = self.state_memory

            stocks = [self.data.tic]
            columns = ["col_{}".format(i) for i in range(len(self.state))]
            columns[self.cash_idx] = "cash"
            for idx, i in enumerate(self.depot_idxs):
                columns[i] = "Shares_{}".format(stocks[idx])
            for idx, i in enumerate(self.price_idxs):
                columns[i] = "Price_{}".format(stocks[idx])
            for idx, i in enumerate(self.buy_price_idxs):
                columns[i] = "BuyPrice_{}".format(stocks[idx])

            df_states = pd.DataFrame(state_list,columns=columns)
            df_states["TotalValue"] = df_states.apply(lambda row: self.calculate_total_asset_value(row), axis=1)
            df_states.index = df_date.date
        # print(df_states)
        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self, normalize=True):
        e = DummyVecEnv([lambda: self])
        if normalize:
            e = VecNormalize(e, norm_obs=True, norm_reward=True, clip_obs=10.)
        obs = e.reset()
        return e, obs
