import highway_env.envs.highway_env as highway_env
import gym
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines import DQN



env = gym.make("highway-v0")

OBS_SHAPE = env.observation_space
ACT_SHAPE = env.action_space
TRAIN_STEP = 1e2


model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)
model.learn(total_timesteps=int(TRAIN_STEP))
model.save("highway")
del model  # delete trained model to demonstrate loading



print("hello")