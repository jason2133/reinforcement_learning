#matplotlib inline

import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict

from blackjack import *
from plotting import *


def mc_prediction(policy, env, num_episodes, discount_factor=1.0):

    # Keeps track of sum and count of returns for each state to calculate an average.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final value function
    V = defaultdict(float)

    ############################
    # YOUR IMPLEMENTATION HERE #

    # for loop for each episode
    
    for i_episode in range(1, num_episodes+1):
        if i_episode % 1000 == 0:
            print(f'Episode {i_episode} / {num_episodes}')
            sys.stdout.flush()
        
        episode = []
        state = env.reset()

        for t in range(100):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        
        states_in_episode = set([tuple(x[0]) for x in episode])
        for state in states_in_episode:
            first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == state)
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
            returns_sum[state] = returns_sum[state] + G
            returns_count[state] = returns_count[state] + 1.0
            V[state] = returns_sum[state] / returns_count[state]

    ############################

    return V

def sample_policy(observation):
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1


# You may change the parameters in the functions below
if __name__ == "__main__":

    matplotlib.style.use('ggplot')
    env = BlackjackEnv()

    V_10k = mc_prediction(sample_policy, env, num_episodes=10000)
    plot_value_function(V_10k, title="Evaluation at 10,000 Steps")

    V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
    plot_value_function(V_500k, title="Evaluation at 500,000 Steps")

