from collections import deque
import numpy as np

import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


class DQNAgent(object):
    def __init__(self, capacity):
        self.D = deque(maxlen=capacity)
        self.weights = np.ra


if __name__ == '__main__':

    env = gym.make('CartPole-v1')

    # Enregistrement de l'Agent
    agent = DQNAgent(env.action_space)

    outdir = 'cartpole-v0/random-agent-results'
    envm = wrappers.Monitor(env,
                            directory=outdir,
                            force=True,
                            video_callable=False)
    env.seed(0)

    episode_count = 1000000
    reward = 0
    done = False
    env.verbose = True
    np.random.seed(5)
    rsum = 0

    for i in range(episode_count):
        obs = envm.reset()
        s1 = 0
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render()
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " +
                      str(j) + " actions")
                break

    print("done")
    env.close()
