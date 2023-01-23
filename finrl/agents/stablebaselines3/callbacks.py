from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import Image, TensorBoardOutputFormat

import tensorflow as tf
import numpy as np

from finrl.plot import plot_actions, plot_states

import logging
logger = logging.getLogger(__name__)

def get_write_checkpoint_cb(freq, log_dir):
    checkpoint_callback = CheckpointCallback(
      save_freq=freq,
      save_path=log_dir,
      name_prefix="rl_model",
      save_replay_buffer=True,
      save_vecnormalize=True,
    )
    return checkpoint_callback


class RenderCallback(BaseCallback):

    def __init__(self, env, log_dir, name, freq=10000, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.eval_env = env
        self.log_dir = log_dir
        self.freq = freq
        self.name = name

        self.n_calls = 0

    def on_training_start(self, locals_, globals_):
        # Those are reference and will be updated automatically
        self.locals = locals_
        self.globals = globals_
        # Update num_timesteps in case training was done before
        self.num_timesteps = self.model.num_timesteps
        self._on_training_start()

    def _on_step(self):

        if self.freq > 0 and self.n_calls % self.freq == 0:

            test_env, test_obs = self.eval_env.get_sb_env()
            """make a prediction"""
            account_memory = []
            actions_memory = []
            # state_memory = []

            test_env.reset()
            for i in range(len(self.eval_env.df.index.unique())):
                action, _states = self.model.predict(test_obs, deterministic=True)
                test_obs, rewards, dones, info = test_env.step(action)

                # Get the memory variables. For some reason they are empty when terminal is reached
                if i == (len(self.eval_env.df.index.unique()) - 2) or i % 10 == 0:
                    state_memory = test_env.env_method(method_name="save_state_memory")
                    actions_memory = test_env.env_method(method_name="save_action_memory")
                #   state_memory=test_env.env_method(method_name="save_state_memory") # add current state to state memory
                # else:
                #     state_memory, actions_memory = [], []
                if dones[0]:
                    print("hit end!")
                    break

            if len(actions_memory)>0:
                try:
                    fig_per_stock = plot_actions(self.eval_env.df, actions_memory[0])
                    self.logger.record("images_{}/per_stock".format(self.name), Image(fig_per_stock, "HWC"), exclude=("stdout", "log", "json", "csv"))
                except Exception as e:
                    logger.error("FAILED TO LOG IMAGES: {}".format(e))
            if len(state_memory) > 0:
                try:
                    fig_states = plot_states(state_memory[0])
                    self.logger.record("images_{}/all".format(self.name), Image(fig_states, "HWC"), exclude=("stdout", "log", "json", "csv"))
                except Exception as e:
                    logger.error("FAILED TO LOG IMAGES: {}".format(e))

            else:
                print("PRINT WAS UNABLAE TO PLOT OMAGES: QALSO LOOK HERE FOR INDEIVIUDLA TB LOGGING")

    def on_step(self) -> bool:
        """
        This method will be called by the model after each call to ``env.step()``.

        For child callback (of an ``EventCallback``), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        self.n_calls += 1
        self.num_timesteps = self.model.num_timesteps

        return self._on_step()



def get_eval_cb(eval_env, log_dir, freq=10000):

    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=freq,
                                 deterministic=True, render=False)


    return eval_callback


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:

        # Get the SummaryWriter for logging histograms
        sw = [of for of in self.logger.output_formats if isinstance(of, TensorBoardOutputFormat)][0].writer

        # Log weights
        if self.locals["dones"][0] and False:
            weights = self.model.get_parameters()["policy"]
            for layer, w_and_b in weights.items():
                sw.add_histogram('weights/{}'.format(layer.replace(".", "/")), w_and_b, self.num_timesteps)

        for k, v in self.locals["infos"][0].items():
            if "images" in k:
                self.logger.record("{}".format(k), Image(v, "HWC"),exclude=("stdout", "log", "json", "csv"))
            elif 'histograms' in k:
                sw.add_histogram('{}'.format(k), np.array(v), self.num_timesteps)
            else:
                self.logger.record(key="{}".format(k), value=v)

        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        except BaseException:
            self.logger.record(key="train/reward", value=self.locals["reward"][0])
        return True
