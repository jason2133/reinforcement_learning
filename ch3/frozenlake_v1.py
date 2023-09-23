import gym
env = gym.make('FrozenLake-v1', render_mode='human')
env.reset()
env.render()