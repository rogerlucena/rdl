import numpy as np


class FeaturesExtractor(object):
    def __init__(self, outSize):
        super().__init__()
        self.outSize = outSize * 3

    def getFeatures(self, obs):
        state = np.zeros((3, np.shape(obs)[0], np.shape(obs)[1]))
        state[0] = np.where(obs == 2, 1, state[0])
        state[1] = np.where(obs == 4, 1, state[1])
        state[2] = np.where(obs == 6, 1, state[2])
        return state.reshape(1, -1)
