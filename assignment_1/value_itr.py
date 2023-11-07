### MDP Value Iteration 
import argparse
import numpy as np
import gymnasium as gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

parser = argparse.ArgumentParser(
    description="A program to run assignment 1 implementations.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--env",
    type=str,
    help="The name of the environment to run your algorithm on: deterministic or stochastic",
    choices=["D-8x8-FrozenLake-v0", "S-8x8-FrozenLake-v0"],
    default="D-8x8-FrozenLake-v0",
)

parser.add_argument(
    "--render-mode",
    "-r",
    type=str,
    help="The render mode: 'human' opens a window to render. 'ansi' does not render anything.",
    choices=["human", "ansi"],
    default="human",
)


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):

    new_policy = np.zeros(nS, dtype="int")

    ############################
    # YOUR IMPLEMENTATION HERE #
    
    for i in range(nS):
        action_reward = []
        for j in range(nA):
            prob, next_s, reward, terminal = P[i][j][0]
            action_reward.append(reward + gamma * prob * value_from_policy[next_s])
        new_policy[i] = np.argmax(action_reward)

    ############################
    return new_policy



def value_iteration(P, nS, nA, gamma=0.9, error=1e-5):

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    ############################
    # YOUR IMPLEMENTATION HERE #
    
    while True:
        new_v_function = np.copy(value_function)
        for s in range(nS):
            action_reward = []
            for a in range(nA):
                prob, next_s, reward, terminal = P[s][a][0]
                action_reward.append(reward + gamma * prob * new_v_function[next_s])
            new_v_function[s] = np.max(action_reward)
        value_change = np.sum(np.abs(value_function - new_v_function))
        value_function = new_v_function
        if value_change < error:
            break
    
    for s in range(nS):
        action_reward = []
        for a in range(nA):
            rob, next_s, reward, terminal = P[s][a][0]
            action_reward.append(reward + gamma * prob * new_v_function[next_s])
        policy[s] = np.argmax(action_reward)

    ############################
    return value_function, policy


def render_single(env, policy, max_steps=100):
    episode_reward = 0
    ob, _ = env.reset()
    for t in range(max_steps):
        env.render()
        time.sleep(0.25)
        a = policy[ob]
        ob, rew, done, _, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    env.render()
    if not done:
        print(
            "Error: your agent cannot reach a terminal state in {} steps.".format(
                max_steps
            )
        )
    else:
        print("Episode reward: %f" % episode_reward)


# You may change the parameters in the functions below
if __name__ == "__main__":
    # read in script argument
    args = parser.parse_args()

    # Make gym environment
    env = gym.make(args.env, render_mode=args.render_mode)

    env.nS = env.nrow * env.ncol
    env.nA = 4

    print("\n" + "-" * 25 + "\nValue Iteration\n" + "-" * 25)
    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.8, error=1e-5)

    render_single(env, p_vi, 100)

