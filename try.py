import highway_env.envs.highway_env as highway_env
import gym
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines import DQN,PPO2
from stable_baselines.common.vec_env import SubprocVecEnv

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init



# env_id = "highway-v0"
# # env = gym.make("highway-v0")

# OBS_SHAPE = env.observation_space
# ACT_SHAPE = env.action_space
# TRAIN_STEP = 1e2


# model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)
# model.learn(total_timesteps=int(TRAIN_STEP))
# model.save("highway")
# del model  # delete trained model to demonstrate loading
# model = DQN.load("highway")

# print("hello")

# while True:
#     obs = env.reset()
#     for i in range(1000):
#         action, _states = model.predict(obs)
#         obs, rewards, dones, info = env.step(action)
#         if dones:
#             break
#         print(obs, rewards, dones, info)
#         env.render()



# MultiProcessing


if __name__ == '__main__':
    env_id = "highway-v0"
    num_cpu = 12  # Number of processes to use
    TRAIN_STEP = 2e3
    # Create the vectorized environment
    # env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = gym.make(env_id)
    env = make_vec_env(env_id, n_envs=num_cpu, seed=0)

    model = PPO2("MlpPolicy", env, verbose=1, tensorboard_log="./",n_steps=16)
    model.learn(total_timesteps=int(TRAIN_STEP))
    model.save("highway")
    # model = PPO2.load("highway")
    # obs = env.reset()
    # while True:
    #     obs = env.reset()
    #     for i in range(1000):
    #         action, _states = model.predict(obs)
    #         obs, rewards, dones, info = env.step(action)
    #         if dones:
    #             break
    #         print(obs, rewards, dones, info)
    #         env.render()
