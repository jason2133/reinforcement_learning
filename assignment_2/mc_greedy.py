#matplotlib inline

import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict

from blackjack import *
from plotting import *


def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):

    # Keeps track of sum and count of returns for each state to calculate an average.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final action-value function.
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
 
    ################################################
    # YOUR IMPLEMENTATION HERE 
    #

    for i_episode in range(1, num_episodes + 1):
        if i_episode % 1000 == 0:
            print(f'Episode {i_episode} / {num_episodes}')
            sys.stdout.flush()
        
        episode = []
        state = env.reset()
        
        for t in range(100):
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        
        sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == state and x[1] == action)
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
            returns_sum[sa_pair] = returns_sum[sa_pair] + G
            returns_count[sa_pair] = returns_count[sa_pair] + 1.0
            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]

    ################################################
    
    return Q, policy

#
# You may change the parameters in the functions below
if __name__ == "__main__":
    matplotlib.style.use('ggplot')
    env = BlackjackEnv()

    Q, policy = mc_control_epsilon_greedy(env, num_episodes=10000, epsilon=0.1)
    V = defaultdict(float)
    for state, actions in Q.items():
        action_value = np.max(actions)
        V[state] = action_value
    plot_value_function(V, title="Optimal Value Function")

