#! /usr/bin/env python3

"""
- Notes about first visit MC:
    1. I see scenario where you get stuck in a state, during the state transition
        - We need an extra penalty for getting stuck. RL is too dumb to realize it itself, 
    2. In value iteration, you know the model. So, you can get an accurate estimate of value at each
    state. But here, you may not even get to the succeed state. 
    So, it's important to have penalty
    3. Monte Carlo Control may not converge.
        - History being reversed didn't seem to be a problem?
        - All ice melts in 100steps. But be careful, don't set that penalty too high
        - large initial epsilon and shrinking it seems to be beneficial
        - The more episodes, the better? Not necessarily. You want to make sure your penalty is set well.
            - Start with a minimal map, and see if you get stuck
        - I see oscillation: 
            1. the agent couldn't find a way to the goal using its current optimal path
            2. Then, until max_step is reached, it will give it lower penalty. That's a local maximum in the
                Q(s,a) world
            3. Epsilon-greedy could reinforce this oscillation, as exploration doesn't have enough data
            points to correct it
"""

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
from frozen_lake_env import FrozenLakeEnv
import numpy as np
import gym
import os
import time
from collections import deque, defaultdict
from typing import Tuple, Deque
import matplotlib.pyplot as plt

TOL = 0.001
RUNS_NUM = 10
EPISODE_NUM = 2000
IMPORTANCE_SAMPLING_RUNS_NUM = 10
GAMMA = 0.98
total_rewards = np.zeros((RUNS_NUM, EPISODE_NUM))

np.set_printoptions(precision=3)

#####################################################
# Tool Functions
#####################################################

def epsilon_greedy_policy(policy, state, nA, epsilon):
    """Generate a randomly distributed number. If it is less than epsilon, return a random action, 
    else return the action with the highest policy value
    nA is the number of actions
    """
    if np.random.uniform() < epsilon:
        return np.random.randint(nA)
    else:
        return policy[state]

def render_single(env, policy, state_output_file, max_steps=100):
    """
    env: gym.core.Environment - Environment to play on. Must have nS, nA, and P as attributes.
    Policy: np.array of shape [env.nS]. The action to take at a given state
    """ 
    episode_reward = 0
    ob = env.reset()
    try:
        os.remove(state_output_file)
    except FileNotFoundError:
        pass
    #TODO Remember to remove
    print(f'=========================Single Run========================')
    for t in range(max_steps):
        env.render(state_output_file)
        old_state = ob
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
        #TODO Remember to remove
        print(f"state: {old_state}, action: {a}, reward: {rew}")
    env.render(state_output_file)
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)

def run_one_episode(env, policy, epsilon, state_output_file:str, max_steps:int=100) -> Deque[Tuple]:
    """

    Args:
        env: gym.core.Environment - Environment to play on. 
            Must have nS, nA, and P as attributes.
        Policy: np.array of shape [env.nS]. The action to take at a given state
            Here the policy is deterministic
        state_output_file (str): output file name
        max_steps (int, optional): maximum number of steps per episode. Defaults to 100.

    Returns:
        Deque[Tuple]: replay buffer with [(s, a, r, s')...]
    """
    
    episode_reward = 0
    ob = env.reset()
    episodic_replay_buffer = deque()
    for t in range(max_steps):
        env.render(state_output_file)
        current_state = ob
        a = epsilon_greedy_policy(policy, current_state, env.nA, epsilon)
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        next_state = ob
        # episodic_replay_buffer: [s, a, r, s']
        # we are appending to the beginning, so you don't have to revert it
        episodic_replay_buffer.appendleft((current_state, a, rew, next_state))
        if done:
            break
    env.render(state_output_file)
    return episodic_replay_buffer

def plot_rewards(total_rewards):
    """Plot rewards

    Args:
        total_rewards (list-like): 2D array: [run1[episode1 ...], run2[episode1 ...]...]
    """
    avg_rewards = np.mean(np.asarray(total_rewards), axis=0)
    plt.plot(range(EPISODE_NUM), avg_rewards)
    plt.show()

#####################################################
# Business Logic
#####################################################

def first_visit_mc(env):
    for run_i in range(RUNS_NUM):
        policy = np.zeros(env.nS, dtype=int)
        N = 1
        # 2D array: column is each action, row is each state
        Q_function = np.zeros((env.nS, env.nA))
        # history of G value of Q
        GQ = np.zeros((env.nS, env.nA))
        epsilon = 0.6
        for episode_i in range(EPISODE_NUM):
            # This is equivalent to policy evaluation in MDP. We are learning the value function!
            # Also, we use the updated policy immediately
            episodic_replay_buffer = run_one_episode(env, policy, epsilon, "first_visit_mc.tmp", 100)
            # After one episode, update GQ: GQ = G+gamma * GQ[-1]
            # this is a trick we use: append a zero that we don't care
            gamma_rewards_history = deque([0])
            undiscounted_total_reward = 0
            for transition in episodic_replay_buffer:
                s, a, r, s_prime = transition
                gamma_reward = r + GAMMA * gamma_rewards_history[-1]
                gamma_rewards_history.append(gamma_reward)
                GQ[s][a] = gamma_reward
                undiscounted_total_reward += r
            # Update Q: 
            new_Q_function = Q_function + 1/N * (GQ-Q_function)
            Q_function = new_Q_function
            policy = np.argmax(Q_function, axis=1)
            N += 1
            epsilon *= 0.995
            
            total_rewards[run_i][episode_i] = undiscounted_total_reward
    print(f'Q_function: {Q_function}')
    print(f'policy: {policy}')
    return Q_function, policy

def evaluate_policy_importance_sampling(target_policy, env):
    # behavior is totally random
    behavior_epsilon = 1.0
    behavior_probability = behavior_epsilon/env.nA
    # Oh man, it's technically zero since we like determinstic policy in real life
    target_epislon = 0.2
    print(f'target policy: {target_policy}')
    V_all = []
    for run_i in range(IMPORTANCE_SAMPLING_RUNS_NUM):
        V = np.zeros(env.nS)
        # values of s1 across EPISODE_NUM
        V_s1_history = []
        for episode_i in range(EPISODE_NUM):
            episodic_replay_buffer = run_one_episode(env, policy, behavior_epsilon, 
                                                    "first_visit_mc.tmp", 100)
            episodic_target_G = np.zeros((env.nS, env.nA))
            G = 0
            weight = 1
            for e_pair in episodic_replay_buffer:
                s, a, r, s_prime = e_pair
                G = (r + GAMMA * G)
                target_probability = target_epislon/env.nA +\
                    (1-target_epislon)*(target_policy[s] == a)

                weight *= (target_probability/behavior_probability)
                episodic_target_G[s][a] = weight * G
            # V(s) = mean of weighted G
            V += np.sum(episodic_target_G, axis=1)
            V_tmp = V/(episode_i + 1)
            V_s1_history.append(V_tmp[0])
        V_all.append(V_s1_history)
    print(f'V_all after importance sampling: {V_all}')
    return V_all

def every_visit_mc():
    """
    We are using a uniformly distributed random policy
    """
    policy = np.zeros(env.nS, dtype=int) 
    V = np.zeros(env.nS)
    V_history = np.zeros((EPISODE_NUM, env.nS))
    # N starts from 1
    Ns = np.ones(env.nS)
    for i in range(EPISODE_NUM):
        G = 0.0
        # epsilon = 1.0
        episodic_replay_buffer = run_one_episode(env, policy, 1.0, "first_visit_mc.tmp", 100)
        for transition in episodic_replay_buffer:
            s, a, r, s_prime = transition
            G = r + GAMMA * G
            # every time we visit s, update with 1/N
            V[s] = V[s] + G
        V = V/Ns
        V_history[i] = V.copy()
    return V_history
    
envs = [
    FrozenLakeEnv(map_name="4x4", is_slippery=False), 
    # FrozenLakeEnv(map_name="8x8", is_slippery=False), 
    # FrozenLakeEnv(map_name="4x4", is_slippery=True)
    ]
for env in envs:
    # first visit mc
    # Q_function, policy = first_visit_mc(env)
    # V_all = evaluate_policy_importance_sampling(policy, env)
    # plot_rewards(V_all)
    # render_single(env, policy, "2_frozen_lake_monte_carlo.tmp", 100)

    # every visit mc
    every_visit_mc()