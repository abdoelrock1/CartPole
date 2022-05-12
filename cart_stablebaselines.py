import gym
# import pybullet, pybullet_envs
import torch as th

from stable_baselines3.dqn.dqn import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


# Create environment
env = gym.make('CartPole-v1')

policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[32, 16])
# Instantiate the agent
model = DQN('MlpPolicy', env,learning_rate=0.0001,policy_kwargs=policy_kwargs, verbose=0)
# Train the agent
for i in range(1000):
    print("Training itteration ",i)
    model.learn(total_timesteps=100000)
    # Save the agent
    model.save("Cart_Pole")
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=5)
    print("mean_reward ", mean_reward)
    if mean_reward >= 100:
        print("***Agent Trained with average reward ", mean_reward)
        break