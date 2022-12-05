"""

Part 1. Problem Definition

This problem is to design an automated trading solution for single stock trading. We model the stock trading process as a Markov Decision Process (MDP). We then formulate our trading goal as a maximization problem.

The algorithm is trained using Deep Reinforcement Learning (DRL) algorithms and the components of the reinforcement learning environment are:

* Action: The action space describes the allowed actions that the agent interacts with the
environment. Normally, a ∈ A includes three actions: a ∈ {−1, 0, 1}, where −1, 0, 1 represent
selling, holding, and buying one stock. Also, an action can be carried upon multiple shares. We use
an action space {−k, ..., −1, 0, 1, ..., k}, where k denotes the number of shares. For example, "Buy
10 shares of AAPL" or "Sell 10 shares of AAPL" are 10 or −10, respectively

* Reward function: r(s, a, s′) is the incentive mechanism for an agent to learn a better action. The change of the portfolio value when action a is taken at state s and arriving at new state s',  i.e., r(s, a, s′) = v′ − v, where v′ and v represent the portfolio
values at state s′ and s, respectively

* State: The state space describes the observations that the agent receives from the environment. Just as a human trader needs to analyze various information before executing a trade, so
our trading agent observes many different features to better learn in an interactive environment.

* Environment: Dow 30 consituents


The data of the single stock that we will be using for this case study is obtained from Yahoo Finance API. The data contains Open-High-Low-Close price and volume.
"""
# import libraries
from __future__ import annotations

import datetime
import os
import sys
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import time

from finrl import config
from finrl import config_tickers
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.main import check_and_make_directories
from finrl.meta.data_processor import DataProcessor
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.plot import backtest_plot
from finrl.plot import backtest_stats
from finrl.plot import get_baseline
from finrl.plot import plot_actions

matplotlib.use('TkAgg')
# %matplotlib inline
sys.path.append("../FinRL-Library")

import itertools
import time
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)

def get_last_checkpoint(dir, num_ckpt_idx=2):

    ckpt_files = [f for f in os.listdir(dir) if f.endswith("zip")]
    if len(ckpt_files) == 0:
        print("Could not find a checkpoint in passed agent dir")
        return None
    else:
        max_ckpt = -1
        max_ckpt_file = None
        for i, f in enumerate(ckpt_files):
            if f == "best_model.zip":
                continue
            ckpt_num = int(f.split("_")[num_ckpt_idx])

            if ckpt_num > max_ckpt:
                max_ckpt = ckpt_num
                max_ckpt_file = f

        return max_ckpt_file

MODEL = "PPO"
TRAIN_NEW_AGENT = True
RETRAIN_AGENT = False
TRAINED_AGENT_PATH = "/home/matthias/Projects/FinRL/trained_models/ppo_1669112226/rl_model_740000_steps.zip"  # 490
if TRAIN_NEW_AGENT == True or RETRAIN_AGENT == True:
    assert TRAIN_NEW_AGENT != RETRAIN_AGENT, "Both Training and Retraining are active. Choose one!"

# If passed model dir is not a file, try to find it in dir
if not os.path.isfile(TRAINED_AGENT_PATH):
    if os.path.isdir(TRAINED_AGENT_PATH):
        ckpt_name = get_last_checkpoint(TRAINED_AGENT_PATH)
        if ckpt_name is None:
            print("Did not find a checkpoint in dir '{}'".format(TRAINED_AGENT_PATH))
            exit()
        else:
            TRAINED_AGENT_PATH = os.path.join(TRAINED_AGENT_PATH, ckpt_name)


if not TRAIN_NEW_AGENT:
    print("Using model: {}".format(TRAINED_AGENT_PATH))


DATA = "DOW30"
check_and_make_directories(
    [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]
)

# Part 3. Download Data
# * Call Limit: Using the Public API (without authentication), you are limited to 2,000 requests per hour per IP
# (or up to a total of 48,000 requests a day).
#
# # -----
# class YahooDownloader:
#     Provides methods for retrieving daily stock data from
#     Yahoo Finance API
#
#     Attributes
#     ----------
#         start_date : str
#             start date of the data (modified from config.py)
#         end_date : str
#             end date of the data (modified from config.py)
#         ticker_list : list
#             a list of stock tickers (modified from config.py)
#
#     Methods
#     -------
#     fetch_data()
#         Fetches data from yahoo API


if DATA == "DOW30":
    tickers = config_tickers.DOW_30_TICKER
else:
    print("Didnt recognize specified data '{}'".format(DATA))
    exit()
data_file = os.path.join("datasets", '{}'.format(DATA))


def preprocess_data(df):
    # """
    # # Part 4: Preprocess Data
    # Data preprocessing is a crucial step for training a high quality machine learning model. We need to check for missing data and do feature engineering in order to convert the data into a model-ready state.
    # * Add technical indicators. In practical trading, various information needs to be taken into account, for example the historical stock prices, current holding shares, technical indicators, etc. In this article, we demonstrate two trend-following technical indicators: MACD and RSI.
    # * Add turbulence index. Risk-aversion reflects whether an investor will choose to preserve the capital. It also influences one's trading strategy when facing different market volatility level. To control the risk in a worst-case scenario, such as financial crisis of 2007–2008, FinRL employs the financial turbulence index that measures extreme asset price fluctuation.
    # """

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.INDICATORS,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False,
    )

    processed = fe.preprocess_data(df)

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed["date"].min(), processed["date"].max()).astype(str))
    combination = list(itertools.product(list_date, list_ticker))

    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"], how="left")
    processed_full = processed_full[processed_full["date"].isin(processed["date"])]
    processed_full = processed_full.sort_values(["date", "tic"])

    processed_full = processed_full.fillna(0)
    processed_full.sort_values(["date", "tic"], ignore_index=True).head(10)

    return processed_full


if os.path.isfile(data_file):
    df = pd.read_pickle(data_file)
    # Check if Update is neccessary
    last_date = df.iloc[-1]['date']
    last_date = datetime.datetime.strptime(last_date, "%Y-%m-%d")
    final_target_date = datetime.datetime.strptime(TRADE_END_DATE, "%Y-%m-%d")
    diff = (final_target_date - last_date).days

    if diff > 4:
        print("Not updated after changing to save preprocessed dtaframe!")
        exit()
        print("Doing data update (LastDate: '{}', Target date: '{}')".format(last_date, final_target_date))
        update_df = YahooDownloader(
            start_date=last_date.strftime("%Y-%m-%d"),
            end_date=final_target_date.strftime("%Y-%m-%d"),
            ticker_list=tickers,
        ).fetch_data()
        df = pd.concat(df, update_df)
        print("Last date of updated df: '{}'".format(df.iloc[-1]['date']))
        df.to_pickle(data_file)
    else:
        print("Loaded PREPROCESSED! dataframe '{}'".format(data_file))

else:
    print("Creating new PREPROCESED! dataframe '{}'".format(data_file))
    df = YahooDownloader(
        start_date=TRAIN_START_DATE,
        end_date=TRADE_END_DATE,
        ticker_list=tickers,
    ).fetch_data()

    df = preprocess_data(df)

    df.to_pickle(data_file)

print(f"config_tickers.DOW_30_TICKER: {config_tickers.DOW_30_TICKER}")
print(f"df.shape: {df.shape}")
df.sort_values(["date", "tic"], ignore_index=True).head()

# """
# # Part 5. Design Environment
# Considering the stochastic and interactive nature of the automated stock trading tasks, a financial task is modeled
# as a **Markov Decision Process (MDP)** problem. The training process involves observing stock price change, taking an
# action and reward's calculation to have the agent adjusting its strategy accordingly. By interacting with the
# environment, the trading agent will derive a trading strategy with the maximized rewards as time proceeds.
# Our trading environments, based on OpenAI Gym framework, simulate live stock markets with real market data according
# to the principle of time-driven simulation.
#
# The action space describes the allowed actions that the agent interacts with the environment. Normally, action a
# includes three actions: {-1, 0, 1}, where -1, 0, 1 represent selling, holding, and buying one share. Also, an action
# can be carried upon multiple shares. We use an action space {-k,…,-1, 0, 1, …, k}, where k denotes the number of
# shares to buy and -k denotes the number of shares to sell. For example, "Buy 10 shares of AAPL" or "Sell 10 shares
# of AAPL" are 10 or -10, respectively. The continuous action space needs to be normalized to [-1, 1], since the policy
# is defined on a Gaussian distribution, which needs to be normalized and symmetric.
#
# # Training data split: 2009-01-01 to 2020-07-01
# # Trade data split: 2020-07-01 to 2021-10-31
# """

train = data_split(df, TRAIN_START_DATE, TRAIN_END_DATE)
test = data_split(df, TEST_START_DATE, TEST_END_DATE)
trade = data_split(df, TRADE_START_DATE, TRADE_END_DATE)
print(f"len(train): {len(train)}")
print(f"len(train): {len(test)}")
print(f"len(trade): {len(trade)}")
print(f"train.tail(): {train.tail(3)}")
print(f"test.head(): {test.head(3)}")
print(f"test.tail(): {train.tail(3)}")
print(f"trade.head(): {trade.head(3)}")
print(f"config.INDICATORS: {config.INDICATORS}")

stock_dimension = len(train.tic.unique())
state_space = 1 + 2 * stock_dimension + len(config.INDICATORS) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": config.INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
}

# Turbulence indicators for trading
data_risk_indicator = df[(df.date < "2020-07-01") & (df.date >= "2009-01-01")]
insample_risk_indicator = data_risk_indicator.drop_duplicates(subset=["date"])
insample_risk_indicator.vix.describe()
insample_risk_indicator.vix.quantile(0.996)
insample_risk_indicator.turbulence.describe()
insample_risk_indicator.turbulence.quantile(0.996)

e_train_gym = StockTradingEnv(df=train, **env_kwargs)
e_test_gym = StockTradingEnv(df=test, **env_kwargs)
e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=70, risk_indicator_col="vix", make_plots=True, **env_kwargs)

## Environment for Training
env_train, _ = e_train_gym.get_sb_env()
print(f"type(env_train): {type(env_train)}")
## Environment for Testing
env_test, _ = e_test_gym.get_sb_env()
print(f"type(env_train): {type(env_test)}")

agent = DRLAgent(env=env_train)
train_start_time_str = time.strftime("%Y%m%d-%H%M%S")
# """
# Model Training: 5 models, A2C DDPG, PPO, TD3, SAC
# """
#
# ### Model 1: A2C
# print("Training A2C")
# agent = DRLAgent(env=env_train)
# model_a2c = agent.get_model("a2c")
# trained_a2c = agent.train_model(
#     model=model_a2c, tb_log_name="a2c", total_timesteps=50000
# )
#
#
# ### Model 2: DDPG
# print("Training DDPG")
# agent = DRLAgent(env=env_train)
# model_ddpg = agent.get_model("ddpg")f
# trained_ddpg = agent.train_model(
#     model=model_ddpg, tb_log_name="ddpg", total_timesteps=50000
# )

### Model 3: PPO
if MODEL == "PPO":
    print("Training/Loading PPO")
    agent = DRLAgent(env=env_train)
    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 128,
    }

    model_ppo = agent.get_model("ppo", policy="MlpPolicy", model_kwargs=PPO_PARAMS, tensorboard_log=TENSORBOARD_LOG_DIR)
    if RETRAIN_AGENT:
        model_ppo = model_ppo.load(TRAINED_AGENT_PATH)
        model_ppo.set_env(env_train)
        print("Loaded model for retraining")

    print(model_ppo)
    if TRAIN_NEW_AGENT:
        time_int = int(time.time())
        TRAINED_MODEL_DIR = os.path.join(TRAINED_MODEL_DIR, 'ppo_test_{}'.format(time_int))
        if not os.path.isdir(TRAINED_MODEL_DIR):
            os.mkdir(TRAINED_MODEL_DIR)
        tb_model_name = 'ppo_{}'.format(time_int)

    elif RETRAIN_AGENT:
        TRAINED_MODEL_DIR = os.path.dirname(TRAINED_AGENT_PATH)
        tb_model_name = TRAINED_MODEL_DIR.split(os.sep)[-1]


    if TRAIN_NEW_AGENT or RETRAIN_AGENT:
        trained_ppo = agent.train_model(model=model_ppo,
                                        tb_log_name=tb_model_name,
                                        total_timesteps=100000,
                                        model_dir=TRAINED_MODEL_DIR,
                                        test_env=e_test_gym,
                                        reset_timesteps=TRAIN_NEW_AGENT)  # 50000  1000000
        trained_ppo.save(os.path.join("trained_models", 'PPO_{}.zip'.format(train_start_time_str)))
    else:
        trained_ppo = model_ppo.load(TRAINED_AGENT_PATH)
        print("Loaded model: {}".format(TRAINED_AGENT_PATH))

#
#
# ### Model 4: TD3
# print("Training TD3")
# agent = DRLAgent(env=env_train)
# TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
# model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS)
#
# trained_td3 = agent.train_model(
#     model=model_td3, tb_log_name="td3", total_timesteps=30000
# )
#
#
# ### Model 5: SAC
# print("Training SAC")
# agent = DRLAgent(env=env_train)
# SAC_PARAMS = {
#     "batch_size": 128,
#     "buffer_size": 1000000,
#     "learning_rate": 0.0001,
#     "learning_starts": 100,
#     "ent_coef": "auto_0.1",
# }
# model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)  # changed policy!"! policy="LstmPolicy",
# trained_sac = agent.train_model(
#     model=model_sac, tb_log_name="sac", total_timesteps=60000
# )
# trained_sac.save(os.path.join("trained_models", 'SAC_{}.zip'.format(train_start_time_str)))

# """
# ## Trading
# Assume that we have $1,000,000 initial capital at 2020-07-01. We use the DDPG model to trade Dow jones 30 stocks.
# ### Set turbulence threshold
# Set the turbulence threshold to be greater than the maximum of insample turbulence data, if current turbulence index is greater than the threshold, then we assume that the current market is volatile
# """



# """
# ### Trade
#
# DRL model needs to update periodically in order to take full advantage of the data, ideally we need to retrain our
# model yearly, quarterly, or monthly. We also need to tune the parameters along the way, in this notebook I only use
# the in-sample data from 2009-01 to 2020-07 to tune the parameters once, so there is some alpha decay here as the
# length of trade date extends.
#
# Numerous hyperparameters – e.g. the learning rate, the total number of samples to train on – influence the
# learning process and are usually determined by testing some variations.
#
# """

# trade = data_split(processed_full, '2020-07-01','2021-10-31')

# env_trade, obs_trade = e_trade_gym.get_sb_env()

print(f"trade.head(): {trade.head()}")
df_account_value, df_actions = DRLAgent.DRL_prediction(model=trained_ppo, environment=e_trade_gym)

print(f"df_account_value.shape: {df_account_value.shape}")
print(f"df_account_value.tail(): {df_account_value.tail()}")
print(f"df_actions.head(): {df_actions.head()}")

plot_actions(trade, df_actions)

# """
# # # Part 7: Backtest Our Strategy
# Backtesting plays a key role in evaluating the performance of a trading strategy. Automated backtesting tool is preferred because it reduces the human error. We usually use the Quantopian pyfolio package to backtest our trading strategies. It is easy to use and consists of various individual plots that provide a comprehensive image of the performance of a trading strategy.

# """
# # 7.1 BackTestStats
# pass in df_account_value, this information is stored in env class
# """

# %%

print("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")

perf_stats_all = backtest_stats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)
perf_stats_all.to_csv("./" + config.RESULTS_DIR + "/perf_stats_all_" + now + ".csv")

# %%

# baseline stats
print("==============Get Baseline Stats===========")
baseline_df = get_baseline(
    ticker="^DJI",
    start=df_account_value.loc[0, "date"],
    end=df_account_value.loc[len(df_account_value) - 1, "date"],
)

stats = backtest_stats(baseline_df, value_col_name="close")

# %%

df_account_value.loc[0, "date"]

# %%
df_account_value.loc[len(df_account_value) - 1, "date"]

# %% md
#
# <a id='6.2'></a>
# ## 7.2 BackTestPlot

# %%

print("==============Compare to DJIA===========")
# %matplotlib inline
# S&P 500: ^GSPC
# Dow Jones Index: ^DJI
# NASDAQ 100: ^NDX
backtest_plot(
    df_account_value,
    baseline_ticker="^DJI",
    baseline_start=df_account_value.loc[0, "date"],
    baseline_end=df_account_value.loc[len(df_account_value) - 1, "date"],
)
plt.show()