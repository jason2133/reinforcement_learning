### MDP Policy Iteration
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
    help="The name of the environment to run your algorithm on.",
    choices=["D-8x8-FrozenLake-v0", "S-4x4-FrozenLake-v0"],
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


def policy_evaluation(P, nS, nA, policy, gamma=0.9, error=1e-3):

    value_function = np.zeros(nS)

    ############################
    # YOUR IMPLEMENTATION HERE #

    while True:
        delta = 0
        for state in range(nS):
            old_value = value_function[state]
            action = policy[state]
            value_function[state] = 0
            for i in range(len(P[state][action])):
                prob, nextstate, reward, terminal = P[state][action][i]
                next_value = prob * (reward + gamma * value_function[nextstate])
                value_function[state] = value_function[state] + next_value
            delta = max(delta, abs(old_value - value_function[state]))
        if delta < error:
            break

    ############################
    return value_function


def policy_improvement(env, P, nS, nA, value_from_policy, policy, gamma=0.9):

    new_policy = np.zeros(nS, dtype="int")

    ############################
    # YOUR IMPLEMENTATION HERE #

    for state in range(nS):
        best_action = 0
        best_value= -10000
        for action in range(nA):
            value = 0
            for i in range(len(P[state][action])):
                prob, nextstate, reward, terminal = P[state][action][i]
                value += prob * (reward + gamma * value_from_policy[nextstate])
            if value > best_value:
                best_value = value
                best_action = action
        new_policy[state] = best_action
    
    ############################
    return new_policy


def policy_iteration(env, P, nS, nA, gamma=0.9, error=1e-5):

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    ############################
    # YOUR IMPLEMENTATION HERE #

    prev_policy = None
    i=0
    while i==0 or np.linalg.norm(policy - prev_policy,1)> 0:
        i+=1
        prev_policy = policy
        value_function = policy_evaluation(P, nS, nA, prev_policy, gamma, error)
        policy = policy_improvement(env, P, nS, nA, value_function, prev_policy, gamma)

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

    print("\n" + "-" * 25 + "\nPolicy Iteration\n" + "-" * 25)
    V_pi, p_pi = policy_iteration(env, env.P, env.nS, env.nA, gamma=0.8, error=1e-10)

    render_single(env, p_pi, 100)

