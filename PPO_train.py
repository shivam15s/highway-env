# from ast import NodeTransformer
import highway_env.envs.highway_env as highway_env
import gym
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines import DQN,PPO2
from stable_baselines.common.vec_env import SubprocVecEnv

from torch.utils.tensorboard import SummaryWriter

logdir=  "./run"
file_writer = SummaryWriter(log_dir=logdir)
# file_writer.set_as_default()

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

import os
from tqdm import trange
def train(env_id = "highway-v0",num_cpu=4,log_dir=None,n_steps=1e3,log_step=100):

    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    model = PPO2("MlpPolicy", env, verbose=1, n_steps=16)

    for i in trange(int(n_steps//log_step)):
        model.learn(total_timesteps=int(log_step))
        model.save(os.path.join(logdir,f"highway_{i}"))

        env1 = gym.make(env_id)
        model1 = PPO2.load(os.path.join(logdir,f"highway_{i}"))
        obs = env1.reset()
        net_reward = 0
        for j in range(1000):
            action, _states = model1.predict(obs)
            # print("Action:",action)
            obs, rewards, dones, info = env1.step(action)
            net_reward += rewards
            print("rewards")
            env1.render()
            if dones:
                file_writer.add_scalar('Episode Reward', net_reward, i*log_step)
                file_writer.add_scalar('Episode Length', j, i*log_step)
                break
            
        
        del env1,model1

if __name__ == '__main__':   
    train()