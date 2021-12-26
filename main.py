import gym
import numpy as np

from agent import SARSAAgent
from utils import show_success_rate

EPSILON = 0.2
ALPHA = 0.3
GAMMA = 0.99
EPISODES = 20000
ENV = 'FrozenLake-v0'


def main():
    env = gym.make(ENV)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    agent = SARSAAgent(n_states, n_actions, GAMMA)
    episode_rewards = np.zeros((EPISODES))

    for e in range(EPISODES):
        state = env.reset()
        done = False
        epsilon = EPSILON*(1-e/EPISODES)
        alpha = ALPHA*e/EPISODES
        action = agent.take_action(state, epsilon)
        while not done:
            next_state, reward, done, info = env.step(action)
            next_action = agent.take_action(next_state, epsilon)
            agent.learn(state, action, reward, next_state, next_action, alpha)
            state = next_state
            action = next_action
            episode_rewards[e] += reward

        if(e % 1000 == 0 and e != 0):
            print(f'{e}/{EPISODES} episode completed')

    return episode_rewards


if __name__ == '__main__':
    episode_rewards = main()
    show_success_rate(episode_rewards)
