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

class PolicyIterationAgent(object):
    """Policy Iteration agent!"""

    def __init__(self, action_space):
        self.action_space = action_space
        # print('action_space: ', action_space) # "Discrete(4)" for example"
        self.sizeOfActionSpace = action_space.n
        print('n of action_space: ', self.sizeOfActionSpace) # how to get the size of the action space
        self.vPrev = self.vCurr = np.random.rand(self.sizeOfActionSpace) # previous and current versions of the Value function "v"
        self.piPrev = self.piCurr = {} # previous and current versions of the policy "pi"

    def fit(self, statedic, transitions, epsilon):
        for state, _ in statedic:
            for a in self.action_space:
                self.piPrev[state][a] = np.random.rand()
                self.piCurr[state][a] = np.random.rand()
        
        piChanged = True

        while piChanged:
            self.vPrev = np.random.rand(self.sizeOfActionSpace)
            self.vCurr = np.zeros(self.sizeOfActionSpace)

            while True: # "do ... while" in Python
                # calculate vCurr

                if isCloseEnough(a, b, epsilon):
                    break
            
            # update pi and verify if something changed on it
            

        return 0

    def act(self, observation, reward, done):
        return self.action_space.sample()

class ValueIterationAgent(object):
    """Value Iteration agent!"""

    def __init__(self, action_space):
        self.action_space = action_space
        
    def fit(self, statedic, transitions, epsilon):
        return 0

    def act(self, observation, reward, done):
        return self.action_space.sample()

# Tools

# Verify if two numpy arrays are close enough, comparing elemnts one-on-one (max norm)
def isCloseEnough(a, b, epsilon):
    return not(False in np.isclose(a, b, atol=epsilon))


if __name__ == '__main__':
    env = gym.make("gridworld-v0")
    env.seed(0)  # Initialise le seed du pseudo-random
    print(env.action_space)  # Quelles sont les actions possibles
    print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    env.render()  # permet de visualiser la grille du jeu 
    env.render(mode="human") #visualisation sur la console
    statedic, mdp = env.getMDP()  # recupere le mdp : statedic
    print("Nombre d'etats : ",len(statedic))  # nombre d'etats ,statedic : etat-> numero de l'etat
    state, transitions = list(mdp.items())[0]
    print('state:\n', state)  # un etat du mdp
    print('transitions:\n', transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}

    # Execution avec un Agent
    agent = RandomAgent(env.action_space)

    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.seed()  # Initialiser le pseudo aleatoire
    episode_count = 1 # 1000
    reward = 0
    done = False
    rsum = 0
    FPS = 0.0001
    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render(FPS)
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()