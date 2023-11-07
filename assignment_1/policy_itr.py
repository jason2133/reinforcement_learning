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
        new_v_function = np.copy(value_function)
        for s, a in enumerate(policy):
            prob, next_s, reward, terminal = P[s][a][0]
            new_v_function[s] = reward + gamma * prob * value_function[next_s]
        value_change = np.sum(np.abs(value_function - new_v_function))
        value_function = new_v_function
        if value_change < error:
            break
    
    ############################
    return value_function


def policy_improvement(env, P, nS, nA, value_from_policy, policy, gamma=0.9):

    new_policy = np.zeros(nS, dtype="int")

    ############################
    # YOUR IMPLEMENTATION HERE #

    for s in range(nS):
        action_reward = []
        for a in range(nA):
            prob, next_s, reward, terminal = P[s][a][0]
            action_reward.append(reward + gamma * prob * value_from_policy[next_s])
        new_policy[s] = np.argmax(action_reward)
    
    ############################
    return new_policy


def policy_iteration(env, P, nS, nA, gamma=0.9, error=1e-5):

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    ############################
    # YOUR IMPLEMENTATION HERE #

    while True:
        value_function = policy_evaluation(P, policy, value_function, gamma, error)
        new_policy = policy_improvement(P, nS, nA, value_function, policy, gamma)
        policy_change = (new_policy != policy).sum()
        policy = new_policy
        if policy_change == 0:
            break
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
    V_pi, p_pi = policy_iteration(env, env.P, env.nS, env.nA, gamma=0.8, error=1e-5)

    render_single(env, p_pi, 100)

