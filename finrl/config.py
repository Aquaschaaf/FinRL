# directory
from __future__ import annotations
import os

# ============================================================================
# ============================ Data ==========================================
# ============================================================================
TICKERS = "SINGLE"
# TICKERS = "CUSTOM"
# TICKERS = "DOW_30"
DATA_INTERVAL = "1d"
RAW_DATA_BASE_DIR = "datasets/raw"
PREPRO_DATA_BASE_DIR = "datasets/preprocessed"
RAW_DATA_SAVE_DIR = os.path.join(RAW_DATA_BASE_DIR, DATA_INTERVAL)
PREPRO_DATA_SAVE_DIR = os.path.join(PREPRO_DATA_BASE_DIR, DATA_INTERVAL)
# stockstats technical indicator column names (check https://pypi.org/project/stockstats/ for different names)
INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",  # Commented for testing reasons - Takes long to compute. Uncomment agian!
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]
NORMALIZATION_WINDOW = 10
CLEAN_STRATEGY = 'remove_cols' if not "m" in DATA_INTERVAL else "remove_rows"
USE_VIX = True
USE_TURBULENCE = True if not "m" in DATA_INTERVAL else False

# ============================================================================
# ============================ Dates =========================================
# ============================================================================
# SPLIT_AUTOMATIC = True
# TEST_FRACTION = 0.15
# VAL_FRACTION = 0.15

TRAIN_START_DATE = "2014-01-06"  # bug fix: set Monday right, start date set 2014-01-01 ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 1658 and the array at index 1 has size 1657
TRAIN_END_DATE = "2020-07-31"
TEST_START_DATE = "2020-08-01"
TEST_END_DATE = "2021-10-01"
TRADE_START_DATE = "2021-11-01"
TRADE_END_DATE = "2022-12-14"
# TRAIN_START_DATE = "2022-10-24"  # bug fix: set Monday right, start date set 2014-01-01 ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 1658 and the array at index 1 has size 1657
# TRAIN_END_DATE = "2022-11-24"
# TEST_START_DATE = "2022-11-25"
# TEST_END_DATE = "2022-12-01"
# TRADE_START_DATE = "2022-12-02"
# TRADE_END_DATE = "2022-12-13"


# ============================================================================
# ============================ Environment ===================================
# ============================================================================
ENV_HMAX = 100
ENV_INIT_AMNT = 10000  # 1000000
ENV_REWARD_SCALE = 1.0 # 0.01 # 1e-4

# ============================================================================
# ============================ Model =========================================
# ============================================================================
TRAINED_MODEL_DIR = "trained_models"
TENSORBOARD_LOG_DIR = "tensorboard_log"
RESULTS_DIR = "results"

TEST = True
MODEL = "ppo"
TRAIN_NEW_AGENT = True
RETRAIN_AGENT = False
TRAINED_AGENT_PATH = "/home/matthias/Projects/FinRL/trained_models/ppo_1671279017_SINGLE_1d/rl_model_850000_steps.zip"
TRAIN_TIMESTEPS = 750000 # 1000000 #  750000


# Model Parameters
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 128,
}

DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}
ERL_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": 512,
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 64,  # bug fix:KeyError: 'eval_times' line 68, in get_model model.eval_times = model_kwargs["eval_times"]
}
RLlib_PARAMS = {"lr": 5e-5, "train_batch_size": 500, "gamma": 0.99}


# ============================================================================
# ============================ Timezones =====================================
# ============================================================================
TIME_ZONE_SHANGHAI = "Asia/Shanghai"  # Hang Seng HSI, SSE, CSI
TIME_ZONE_USEASTERN = "US/Eastern"  # Dow, Nasdaq, SP
TIME_ZONE_PARIS = "Europe/Paris"  # CAC,
TIME_ZONE_BERLIN = "Europe/Berlin"  # DAX, TECDAX, MDAX, SDAX
TIME_ZONE_JAKARTA = "Asia/Jakarta"  # LQ45
TIME_ZONE_SELFDEFINED = "xxx"  # If neither of the above is your time zone, you should define it, and set USE_TIME_ZONE_SELFDEFINED 1.
USE_TIME_ZONE_SELFDEFINED = 0  # 0 (default) or 1 (use the self defined)

# ============================================================================
# ============================ Summarys ======================================
# ============================================================================
MODEL_PARAMS = {"ppo":PPO_PARAMS, "a2c": A2C_PARAMS, "sac": SAC_PARAMS}
DIRS = [RAW_DATA_SAVE_DIR, PREPRO_DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]
