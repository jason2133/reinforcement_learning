import gym
env = gym.make('CartPole-v1', render_mode='human')
env.reset()
env.render()
# env.close()
