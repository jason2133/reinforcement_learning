import gym
env = gym.make('CartPole-v1', render_mode='human')
env.reset()

for i in range(100):
    env.step(env.action_space.sample())
    env.render()
env.close()

