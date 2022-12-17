from __future__ import annotations

from typing import List

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from finrl.plot import plot_actions
from finrl.environment.env_stock_trading.Broker import Broker
from stable_baselines3.common.vec_env import DummyVecEnv
import logging

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

        self.cash_idx = 0
        self.price_idxs = np.array(range(1, self.stock_dim + 1))
        self.depot_idxs = np.array(range(self.stock_dim + 1, self.stock_dim * 2 + 1))
        self.buy_price_idxs = np.array(range(self.stock_dim * 2 + 1, self.stock_dim * 3 + 1))
        self.tech_indicator_start_idx = self.stock_dim * 3 + 1
        self.broker = Broker(cash_idx=self.cash_idx,
                             price_idxs=self.price_idxs,
                             buy_price_idxs=self.buy_price_idxs,
                             depot_idxs=self.depot_idxs,
                             stock_dim=self.stock_dim,
                             transaction_cost=0.03)

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

                actions[index], depot, buy_prices, cost = self.broker.sell_stock(index, actions[index], self.state)
                self.cost += cost
                self.state[self.depot_idxs] = depot
                self.state[self.buy_price_idxs] = buy_prices
                if abs(actions[index]) > 0:
                    self.trades += 1

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                actions[index], depot, buy_prices, cost = self.broker.buy_stock(index, actions[index], self.state)
                self.state[self.depot_idxs] = depot
                self.state[self.buy_price_idxs] = buy_prices
                self.cost += cost
                if abs(actions[index]) > 0:
                    self.trades += 1

        return actions


    def calculate_total_asset_value(self):

        cash = self.state[self.cash_idx]
        prices = np.array(self.state[self.price_idxs])
        depot = np.array(self.state[self.depot_idxs])
        depot_value = sum(prices * depot)

        return cash + depot_value



    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def step(self, actions):

        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # If done
        if self.terminal:
            # print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                self.state[0]
                + sum(
                    np.array(self.state[1 : (self.stock_dim + 1)])
                    * np.array(
                        self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    )
                )
                - self.asset_memory[0]
            )  # initial_amount is only cash part of our initial asset
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
                1
            )
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )
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

            return self.state, self.reward, self.terminal, {}

        else:
            # action = Amount of stocks to buy/sell
            actions = actions * self.hmax  # actions initially is scaled between 0 to 1
            actions = actions.astype(int)  # convert into integer because we can't by fraction of shares

            # state = [Cash, Value of Stocks, Num of stocks in Depot, TechnIndicator]
            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)

            begin_total_asset = self.calculate_total_asset_value()

            depot_before = self.state[self.depot_idxs]
            buy_prices_before = self.state[self.buy_price_idxs]
            actions = self._update_depot(actions)
            self.actions_memory.append(actions)
            depot_after = self.state[self.depot_idxs]
            buy_prices_after = self.state[self.buy_price_idxs]

            # Rate sell tranactions -> Sell_price - Buy_price * Amount
            sell_index = np.where(actions < 0)[0]
            trade_reward = 0
            for i in sell_index:
                sell_price = self.state[self.price_idxs][i]
                buy_price = buy_prices_before[i]
                amount = abs(actions[i])
                diff = sell_price - buy_price
                perf = diff * amount

                trade_reward += perf
            trade_reward *= 10
            trade_reward = np.max([trade_reward, 0])
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

            end_total_asset = self.calculate_total_asset_value()

            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())

            self.reward = 0
            # self.reward = end_total_asset - begin_total_asset
            self.reward += trade_reward
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling

            self.state_memory.append(
                self.state
            )  # add current state in state_recorder for each step

        return self.state, self.reward, self.terminal, {}

    def reset(self):
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

        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

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
            df_states = pd.DataFrame(
                state_list,
                columns=[
                    "cash",
                    "Bitcoin_price",
                    "Gold_price",
                    "Bitcoin_num",
                    "Gold_num",
                    "Bitcoin_Disable",
                    "Gold_Disable",
                ],
            )
            df_states.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
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

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
