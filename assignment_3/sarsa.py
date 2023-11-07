import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from cliffwalk import *

def sarsa(env, num_episodes=500, render=True, exploration_rate=0.1,
          learning_rate=0.5, gamma=0.9):
    q_values_sarsa = np.zeros((num_states, num_actions))
    ep_rewards = []
   
    ####################################################
    # YOUR IMPLMENTATION HERE

    # for # of episodes
    for i in range(num_episodes):
        state = env.reset()
        done = False
        reward_sum = 0
        action = egreedy_policy(q_values_sarsa, state, exploration_rate)

        while not done:
            next_state, reward, done = env.step(action)
            reward_sum = reward_sum + reward

            next_action = egreedy_policy(q_values_sarsa, next_state, exploration_rate)
            td_target = reward + gamma * q_values_sarsa[next_state][next_action]
            td_error = td_target - q_values_sarsa[state][action]
            q_values_sarsa[state][action] += learning_rate * td_error

            state = next_state
            action = next_action

            if render:
                env.render(q_values, action=action[action], colorize=True)
        ep_rewards.append(reward_sum)
    ####################################################

    return ep_rewards, q_values_sarsa


# Generating last animation for optimal policy
def play(q_values):
    env = GridWorld()
    state = env.reset()
    done = False

    while not done: 
        action = egreedy_policy(q_values, state, 0.0)
        next_state, reward, done = env.step(action)
        state = next_state
        env.render(q_values=q_values, action=actions[action], colorize_q=True)


#
#
#
# You may change the parameters in the functions below
if __name__ == "__main__":

    env = GridWorld()

    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3
    actions = ['UP', 'DOWN', 'RIGHT', 'LEFT']

    # The number of states in simply the number of "squares" 
    # in our grid world, in this case 4 * 12
    num_states = 4 * 12

    # There are 4 possible actions, up, down, right and left
    num_actions = 4

    q_values_sarsa = np.zeros((num_states, num_actions))

    #
    # for the animation, the procedure of SARSA is shown
    sarsa_rewards, q_values_sarsa = sarsa(env, render=False, learning_rate=0.5, gamma=0.99)
    env.render(q_values_sarsa, colorize_q=True)
    np.mean(sarsa_rewards)

    #
    # for the statistics
    sarsa_rewards, _ = zip(*[sarsa(env, render=False, exploration_rate=0.5) for _ in range(10)])
    avg_rewards = np.mean(sarsa_rewards, axis=0)
    mean_reward = [np.mean(avg_rewards)] * len(avg_rewards)
    fig, ax = plt.subplots()
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Rewards')
    ax.plot(avg_rewards)
    ax.plot(mean_reward, 'g--')

    print('Mean Reward: {}'.format(mean_reward[0]))

    #
    # The animation for the optimal policy found in the first call SARSA is shown
    play(q_values_sarsa)
    xxx=input()
