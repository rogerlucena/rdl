import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import random
from xxhash import xxh64 as hash64
from time import sleep

class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

class QLearning(object):
    """Q-Learning Algorithm"""

    def __init__(self, action_space, epsilon, alpha, gamma):
        self.action_space = action_space
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = {}

    def update(self, obs, obs_old, reward, done, action):
        obs_old_hashed = hash64(obs_old).hexdigest()
        obs_hashed = hash64(obs).hexdigest()

        if not obs_hashed in self.Q:
            self.Q[obs_hashed] = np.zeros(len(self.action_space))

        print('action = ', action)

        if not done:
            self.Q[obs_old_hashed][action] = self.Q[obs_old_hashed][action] + self.alpha * (reward + self.gamma * np.max(self.Q[obs_hashed]) - self.Q[obs_old_hashed][action])
        else:
            self.Q[obs_old_hashed][action] = self.Q[obs_old_hashed][action] + self.alpha * (reward + self.gamma * 0 - self.Q[obs_old_hashed][action])

    def act(self, observation, reward, done):
        #maxa = self.action_space[0]
        #for a in self.action_space:
            #if((s, a))

        obs_hashed = hash64(observation).hexdigest()

        if not obs_hashed in self.Q:
            self.Q[obs_hashed] = np.zeros(len(self.action_space))
        
        a_max = np.argmax(self.Q[obs_hashed])
        if random.random() > self.epsilon:
            return (self.action_space[a_max], a_max)
        else:
            random_int = random.randint(0, len(self.action_space)-2)
            return (self.action_space[random_int], random_int) if random_int < a_max else (self.action_space[random_int + 1], random_int + 1)

        # print('observation:\n', observation)
        # print('type: ', type(observation))


if __name__ == '__main__':
    env = gym.make("gridworld-v0")
    env.seed(0)  # Initialise le seed du pseudo-random
    print(env.action_space)  # Quelles sont les actions possibles
    print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    env.render()  # permet de visualiser la grille du jeu 
    env.render(mode="human") #visualisation sur la console
    # statedic, mdp = env.getMDP()  # recupere le mdp : statedic
    # print("Nombre d'etats : ",len(statedic))  # nombre d'etats ,statedic : etat-> numero de l'etat
    # cstate, transitions = list(mdp.items())[0]
    # cprint(state)  # un etat du mdp
    # print(transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}

    # Execution avec un Agent
    # agent = RandomAgent(env.action_space)
    agent = QLearning(np.arange(env.action_space.n), 0.1, 0.01, 0.99)

    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.seed()  # Initialiser le pseudo aleatoire
    episode_count = 1000
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
            action, action_index = agent.act(obs, reward, done)

            obs_old = obs.copy()

            obs, reward, done, _ = envm.step(action)
            
            agent.update(obs, obs_old, reward, done, action_index)
            
            rsum += reward
            j += 1
            if env.verbose:
                env.render(FPS)
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()