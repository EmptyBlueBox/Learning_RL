from myosuite.utils import gym
env = gym.make('myoChallengeDieReorientP1-v0')
env.reset()
for _ in range(10000):
    env.mj_render()
    env.step(env.action_space.sample())  # take a random action
env.close()
