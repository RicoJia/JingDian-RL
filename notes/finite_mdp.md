## Finite MDP
### Markov Property and Basic Model Setup
Markov property is in short, "All info you need is the present" - . I.e., $P(S_{t+1}|S_{t}) = P(S_{t+1}|S_{t}, S{t-1} ...)$
- "Things have a clear end (terminal state)" is an episode. Your education is episodic, life is continuous.
- A markov decision process is basically a state machine, but with probability to transition to different states
  - And RL is built on top of rewards & transitional probability on a markov decision process

First, we are interested in total rewards. $G_t = R_{t+1}+R_{t+2} ...$ But really, we want to have a diminishing effect on the past rewards. Otherwise. total rewards will blow up!
  - So, we want to discount the total rewards: $G_t=R_{t+1} + \gamma R_{t+2} ... = R_{t+1} + \gamma G_{t+1}$

Second, we define State transitions
  - given the current state and action, there's a probability to transition to different next states
  - we know: $P(s', r| s, a)$ for each current state and action

Third, policy is a look up table of **being in state S, what's the probability of choosing action A**: $\pi (A|S)$ (stochastic). **But you may have determinsistic policy, which is simply a look up table of mapping**: $a \rightarrow s$. 
    - You may have infinite number of stochastic policies, but finite number of deterministic policies

Value Function: **at a given current state s, what would be the total expected reward, starting from this state**
1. Definition
    $$
    v_{\pi}(s) = E_{\pi}[G_t|S_t=s] = E_{\pi}[R_{t+1} + \gamma R_{t+2} + ... |S_t=s]
    \\
    $$
1. Following a given policy $\pi$, at kth iteration,
    $$
    v^{\pi}_k(s) = r(s, \pi(s)) + \gamma \sum_{s'} P(s'|s, \pi(s))V^{\pi}_{k-1}(s')
    $$
    which is the reward following $\pi(s)$ + expectation of all subsequent possible states' values folling $\pi(s')$,
    This is called **Bellman Backup** of a particular policy

1. [Optional] In complex models, you could have multiple $(s', r)$, given $(s,a)$. When going for $s$ to $s'$, ingegrate over $r$, and $s$
    $$
    E_{\pi}[R_{t+1} + \gamma R_{t+2} + ... |S_t=s] =
    \sum_{A} \pi_{A} (a|s) \sum_{r} \sum_{s'}P(s', r|s,a)(r + \gamma E_{\pi}[R_{t+2} + \gamma R_{t+3} + ... |S_{t+1}=s'])
    \\
    = \sum_{A} \pi_{A} (a|s) \sum_{r} \sum_{s'}P(s', r|s,a)(r + \gamma v_{\pi}(s'))
    $$

### Policy Iteration
For simplicity, here we assume we have determinstic policies. and reward is determinstic, given an action pair $(s,a)$
1. Policy Evaluation only: iterate through all policies, find each policie's value function, then pick one that yields highest value. Then, all you need is updating using bellman backup
    1. Initialize all s: $v(s) = 0$
    2. until convergence **at each state**
        $$
        v^{\pi}_k(s) = r(s, \pi(s)) + \gamma \sum_{s'} P(s'|s, \pi(s))V^{\pi}_{k-1}(s')
        $$
    3. Then, optimal policy is the one yields the highest value at every state

1. Policy Iteration: at each step, we do above policy evaluation with $v(s)$, then update $\pi_{k+1}$ using the newly-introduced $Q(s,a)$. This update is a.k.a policy update
    1. Initialize all $v(s)=0$, $\pi(s) = 0$
    1. until convergence in policy at each state
        1. policy evaluation above
            1. Initialize all s: $v(s) = 0$
            2. until convergence **at each state**. We stick to the same policy
                $$
                v^{\pi}_k(s) = r(s, \pi(s)) + \gamma \sum_{s'} P(s'|s, \pi(s))V^{\pi}_{k-1}(s')
                $$
            3. Then, optimal policy is the one yields the highest value at every state
        1. Policy improvement, under current $\pi
            $$
            Q^{\pi}(a, s) = E_{pi}[G_t|s, a] = E_{\pi}[R_{t+1} + \gamma R_{t+2} + ... |S_t=s, A_t=a]
            \\
            = R(s,a) + \gamma \sum_{s'}P(s|s',a)V^{\pi}(s')
            $$
            - Policy Update: when updating $\pi$, we evaluate all actions.
                $$
                \pi_{i+1}(s) = argmax_a Q^{\pi_i}(a,s)
                $$

We can show for a given set of initial values $V(s)$, each iteration its value will only increase (monotonic increase). because you immediately take the optimal policy found in the last step. In this case:
    $$
    v{\pi_{i+1}}(s) = max_{A} q_{\pi}(s,a)
    $$

Once $\pi$ is stable, we don't change anymore. Because subsequent iterations will be the same. Also, there are finite number of iterations, because each policy is only visited once.

  
### Value Iteration
1. Initialize $V0(s)=0$ for all states
1. Loop until $V$ does not change?
    1. For each state $S$,
        $$
        V_{k+1}(s) = max_A R(s,a) + \gamma \sum_S P(s'|s,a)V_k(s')
        $$
        - Note: compared to policy iteration, here in policy evaluation, we choose the best reward among all actions
1. update policy
    $$
    \pi_{k+1}(s) = argmax_a (R(s,a) + \gamma \sum_S P(s'|s,a)V_k(s'))
    $$
    - Note: This is done after V is evaluated

Will Value Iteration converge? Yes.
Say we have $V_{k}(s)$ and $V_{j}(s)$, written in vector for all states. Say $B$ is the Bellman Backup Operator. Can prove $|BV_{k}(s) - BV_J(s)| < |V_{k}(s) - V_{j}(s)|$. Say $j= k+1$, we can see the norm of V decreases.

Q and V:?
$Q = R + \gamma V$
$V_{\pi}^*(s) = argmax_a Q(s,a) = Q(s, \pi^*(s))$


TODO
I'm wondering if there's a guarantee that the value function will converge for a given policy $\pi_i$?

### Assignment
- Great Question: Quetsion 3 of [slides](https://github.com/tallamjr/stanford-cs234/blob/master/assignments/ass1/assignment1_sol.pdf)
- Frozen Lake
- Inverted Pendulum