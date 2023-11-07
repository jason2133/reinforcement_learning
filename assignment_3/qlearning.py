import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from cliffwalk import *

def q_learning(env, num_episodes=500, render=True, exploration_rate=0.1,
               learning_rate=0.5, gamma=0.9):    
    q_values = np.zeros((num_states, num_actions))
    ep_rewards = []
    
    ####################################################
    # YOUR IMPLMENTATION HERE
    
    # for # of episodes
    for i in range(num_episodes):
        state = env.reset()
        done = False
        reward_sum = 0

        while not done:
            action = egreedy_policy(q_values, state, exploration_rate)
            next_state, reward, done = env.step(action)
            reward_sum = reward_sum + reward
            td_target = reward + 0.9 * np.max(q_values[next_state])
            td_error = td_target - q_values[state][action]
            q_values[state][action] += learning_rate * td_error
            state = next_state

            if render:
                env.render(q_values, action=actions[action], colorize_q=True)
    
        ep_rewards.append(reward_sum)
    ####################################################
    
    return ep_rewards, q_values


# Generating last animation for optimal policy
def play(q_values):
    env = GridWorld()
    state = env.reset()
    done = False

    while not done: 
        # Select action
        action = egreedy_policy(q_values, state, 0.0)
        # Do the action
        next_state, reward, done = env.step(action)

        # Update state and action
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
    LEFT =     3
    actions = ['UP', 'DOWN', 'RIGHT', 'LEFT']

    # The number of states in simply the number of "squares" 
    # in our grid world, in this case 4 * 12
    num_states = 4 * 12

    # There are 4 possible actions, up, down, right and left
    num_actions = 4

    q_values = np.zeros((num_states, num_actions))

    #
    # For the animiation, the proceure of Q Learning is shown
    q_learning_rewards, q_values = q_learning(env, num_episodes=100, gamma=0.9, learning_rate=1, render=False)
    env.render(q_values, colorize_q=True)
    np.mean(q_learning_rewards)

    #
    # For the statistics, a bunch of Q Learnings is executed
    q_learning_rewards, _ = zip(*[q_learning(env, render=False, exploration_rate=0.1,
                                         learning_rate=1) for _ in range(100)])
    avg_rewards = np.mean(q_learning_rewards, axis=0)
    # For the graphics of statistics
    mean_reward = [np.mean(avg_rewards)] * len(avg_rewards)
    fig, ax = plt.subplots()
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Rewards')
    ax.plot(avg_rewards)
    ax.plot(mean_reward, 'g--')
    print('Mean Reward: {}'.format(mean_reward[0]))

    # For the enimation for the optimal policy chosen in the first Q-Learning
    play(q_values)
    xxx=input()
