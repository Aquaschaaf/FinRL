import os
import time
import logging

from finrl import config
from finrl import config_tickers
from finrl.agents.stablebaselines3.models import DRLAgent

logger = logging.getLogger(__name__)

class SB3Trainer:

    def __init__(self, train_gym, test_gym):

        self.train_gym = train_gym
        self.train_env, _ = self.train_gym.get_sb_env()
        self.test_gym = test_gym

    def get_model_dirs(self):

        if config.TRAIN_NEW_AGENT:
            time_int = int(time.time())
            if config.TEST:
                model_dir = os.path.join(config.TRAINED_MODEL_DIR, '{}_test_{}_{}_{}_{}'.format(config.MODEL, time_int, config.TICKERS, config.DATA_INTERVAL, config.MODEL_DESCRIPTION))
                tb_model_name = '{}_test_{}_{}_{}_{}'.format(config.MODEL, time_int, config.TICKERS, config.DATA_INTERVAL, config.MODEL_DESCRIPTION)
            else:
                model_dir = os.path.join(config.TRAINED_MODEL_DIR, '{}_{}_{}_{}_{}'.format(config.MODEL, time_int, config.TICKERS, config.DATA_INTERVAL, config.MODEL_DESCRIPTION))
                tb_model_name = '{}_{}_{}_{}_{}'.format(config.MODEL, time_int, config.TICKERS, config.DATA_INTERVAL, config.MODEL_DESCRIPTION)
            if not os.path.isdir(model_dir):
                os.mkdir(model_dir)

        elif config.RETRAIN_AGENT:

            model_dir = os.path.dirname(config.TRAINED_AGENT_PATH)
            tb_model_name = model_dir.split(os.sep)[-1]

        else:
            return None, None

        return model_dir, tb_model_name


    def setup_model_for_retraining(self, model):

        # If passed model dir is not a file, try to find it in dir
        if not os.path.isfile(config.TRAINED_AGENT_PATH):
            if os.path.isdir(config.TRAINED_AGENT_PATH):
                ckpt_name = self.get_last_checkpoint(config.TRAINED_AGENT_PATH)
                if ckpt_name is None:
                    logger.info("Did not find a checkpoint in dir '{}'".format(config.TRAINED_AGENT_PATH))
                    exit()
                else:
                    config.TRAINED_AGENT_PATH = os.path.join(config.TRAINED_AGENT_PATH, ckpt_name)
            else:
                logger.error("Specified Agent '{}' is neither a file nor a directory".format(config.TRAINED_AGENT_PATH))
                exit()
        try:
            model = model.load(config.TRAINED_AGENT_PATH)
        except:
            logger.error("Could not load specified model '{}' for retrianing".format(config.TRAINED_AGENT_PATH))
            exit()
        model.set_env(self.train_env)
        logger.info("Set model '{}' up for retraining".format(config.TRAINED_AGENT_PATH))

        return model

    def get_last_checkpoint(self, dir, num_ckpt_idx=2):

        ckpt_files = [f for f in os.listdir(dir) if f.endswith("zip")]
        if len(ckpt_files) == 0:
            logger.info("Could not find a checkpoint in passed agent dir")
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


    def train(self):
        # Setting up the Agent
        agent = DRLAgent(env=self.train_env)
        model_params = config.MODEL_PARAMS[config.MODEL]

        if config.MODEL == 'recPPO':
            policy = "MlpLstmPolicy"
        else:
            policy = "MlpPolicy"

        model = agent.get_model(config.MODEL, policy=policy, model_kwargs=model_params,
                                tensorboard_log=config.TENSORBOARD_LOG_DIR)
        if config.RETRAIN_AGENT:
            model = self.setup_model_for_retraining(model)

        logger.info(model)
        model_dir, tb_model_name = self.get_model_dirs()
        model = agent.train_model(model=model,
                                  tb_log_name=tb_model_name,
                                  total_timesteps=config.TRAIN_TIMESTEPS,
                                  model_dir=model_dir,
                                  test_gym=self.test_gym,
                                  train_gym=self.train_gym,
                                  reset_timesteps=config.TRAIN_NEW_AGENT)  # 50000  1000000

        model.save(os.path.join(model_dir, 'rl_model_{}_steps.zip'.format(model.num_timesteps)))

        return model

