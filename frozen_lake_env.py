#!/usr/bin/env python3
import numpy as np
import sys
from six import StringIO, b
from contextlib import closing
from gym import utils, Env, spaces
from gym.utils import seeding

################################################################
# Discrete Env
################################################################

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution Each row specifies class probabilities
    Returns an index of prob_n (category)
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.random()).argmax()


class DiscreteEnv(Env):

    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*) 
    {0 current state: {0 (action): [(0.1, 0, 0.0, False), (0.8, 0, 0.0, False), (0.1, 4, 0.0, False)], 1: []......}
    - isd: initial state distribution (**)

    (*) dictionary dict of dicts of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS


    """
    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction=None # for rendering
        self.nS = nS
        self.nA = nA

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self._seed()
        self.reset()
        self._step_num= 0

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # initialize random initial state in int
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction=None
        self._step_num= 0
        return self.s

    def step(self, a: int):
        # a is the index in [0, action_space_length]
        # transitions [(0.1, 0, 0.0, False), (0.8, 0, 0.0, False), (0.1, 4, 0.0, False)]
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d= transitions[i]
        self.s = s
        self.lastaction=a
        # Rico: I added this ice-melting constraint 
        if self._step_num >100:
            r, d = -200, True
        self._step_num += 1 

        return (s, r, d, {"prob" : p})

################################################################
# Frozen Lake Env
################################################################

# Mapping between directions and index number
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Maps for the two different environments
MAPS = {
    # TODO
    # "2x2": [
    #     "SF",
    #     "HG",
    # ], 
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FHFFFHFF",
        "FFFHFFFF",
        "FFHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
}

class FrozenLakeEnv(DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following

        SFFF
        FHFH
        FFFH
        HFFG

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.

    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4",is_slippery=True):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape

        nA = 4 # number of actions
        nS = nrow * ncol # number of states

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col
        def inc(row, col, a):
            if a==0: # left
                col = max(col-1,0)
            elif a==1: # down
                row = min(row+1,nrow-1)
            elif a==2: # right
                col = min(col+1,ncol-1)
            elif a==3: # up
                row = max(row-1,0)
            return (row, col)

        # Adding [probabity, next state, reward, done] to P[state][action]
        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GH':
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a-1)%4, a, (a+1)%4]:
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'GH'
                                if newletter == b'G':
                                    rew = 100
                                elif newletter == b'H':
                                    rew = -100.0
                                else:
                                    if s == newstate:
                                        rew = 0.0
                                    else:
                                        rew = -0.01
                                li.append((0.8 if b==a else 0.1, newstate, rew, done))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'GH'
                            # if we are not going to goal, reward = 0
                            if newletter == b'G':
                                rew = 100.0
                            elif newletter == b'H':
                                rew = -100.0
                            else:
                                rew = 0.0
                                # if s == newstate:
                                #     rew = -0.5
                                # rew = -0.01
                            li.append((1.0, newstate, rew, done))
# Glossary
# P: nested dictionary
# 	From gym.core.Environment
# 	For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
# 	tuple of the form (probability, nextstate, reward, terminal) where
# 		- probability: float
# 			the probability of transitioning from "state" to "nextstate" with "action"
# 		- nextstate: int
# 			denotes the state we transition to (in range [0, nS - 1])
# 		- reward: int
# 			either 0 or 1, the reward for transitioning from "state" to
# 			"nextstate" with "action"
# 		- terminal: bool
# 		  True when "nextstate" is a terminal state (hole or goal), False otherwise
# nS: int
# 	number of states in the environment
# nA: int
# 	number of actions in the environment
# gamma: float
# 	Discount factor. Number in range [0, 1)
# Returns: index of action
        super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)

    def render(self, output_file, mode='human', close=False):
        # This is from the original openAI Gym implementation.
        if close:
            return
        with open(output_file, "a") as outfile:
            row, col = self.s // self.ncol, self.s % self.ncol
            desc = self.desc.tolist()
            desc = [[c.decode('utf-8') for c in line] for line in desc]
            desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
            if self.lastaction is not None:
                outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
            else:
                outfile.write("\n")
            outfile.write("\n".join(''.join(line) for line in desc)+"\n")

