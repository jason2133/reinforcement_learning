import gym
env = gym.make('Pong-v0')
env.reset()
env.render()
env.close()