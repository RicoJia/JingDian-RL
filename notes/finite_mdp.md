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
  - reward could be **different** even when the next state is the same
  - we know: $P(s', r| s, a)$ for each current state and action

So we define
    $$P(S_{t+1}=s'|S_t=s, A_t=a) = \sum_{r} P(s', r|s, a)$$
  - Expected Rewards for a certain state, and action
        $$r(s, a) = E[R_{t+1}|S_t=s, A_t=a] = \sum rP(R_{t+1}=r|s, a) = \sum r \sum_{s'}P(R_{t+1}=r, s'|s, a)$$
  - Expected reward for a certain state, next state, and action (see the first one)
        $$r(s, a, s') = E[R_{t+1}|S_t=s, A_t=a, S_{t+1}=s'] = \sum rP(R_{t+1}=r|s, a, s') = \frac{\sum rP(s', r|s, a)}{P(s'|s,a)}$$
- A (stochastic) policy is a look up table of **being in state S, what's the probability of choosing action A**: $\pi (A|S)$
  - Value Function: **using this lookup table, what would be the total expected reward, starting from current state s**
    - State Value Function, (Estimate by recording average $R$ w.r.t each state visited)
            1. Set up
                $$
                v_{\pi}(s) = E_{\pi}[G_t|S_t=s] = E_{\pi}[R_{t+1} + \gamma R_{t+2} + ... |S_t=s]
                \\
                v_{\pi}(s') = E_{\pi}[G_{t+1}|S_{t+1}=s'] = E_{\pi}[R_{t+2} + \gamma R_{t+3} + ... |S_{t+1}=s']
                \\
                $$
            1. For a given pair of $(s, a, s', r)$, its "update" is
                $$r + \gamma E_{\pi}[R_{t+2} + \gamma R_{t+3} + ... |S_{t+1}=s']$$
            1. But you could have multiple $(s', r)$, given $(s,a)$. When going for $s$ to $s'$, ingegrate over $r$, and $s$
                $$
                E_{\pi}[R_{t+1} + \gamma R_{t+2} + ... |S_t=s] =
                \sum_{A} \pi_{A} (a|s) \sum_{r} \sum_{s'}P(s', r|s,a)(r + \gamma E_{\pi}[R_{t+2} + \gamma R_{t+3} + ... |S_{t+1}=s'])
                \\
                = \sum_{A} \pi_{A} (a|s) \sum_{r} \sum_{s'}P(s', r|s,a)(r + \gamma v_{\pi}(s'))
                $$
            1. $v_{pi}$ is expected $G_t$ at $t$ when state is in $s$. I'm wondering if there's a guarantee that the value function will converge?
    - Action Value Function (Estimate by recording average $R$ w.r.t each states and actions visited)
            $$
            q_{\pi}(a, s) = E_{pi}[G_t|S_t=s, A_t=a] = E_{\pi}[R_{t+1} + \gamma R_{t+2} + ... |S_t=s, A_t=a]
            $$
    - Monte Carlo method: above estimation methods

  - Bellman Equation: TODO
  - Optimal Value: $v_{*}(s) = max_{A} q_{\pi*}(a,s)$, notice the `*` for optimality
    - You choose the max value of the next state + reward
            $$
            v_{*}(s) = max_{A} q_{\pi}(s,a)
            \\
            =max_{A} \sum_{r} \sum_{s} P[s'r|s,a](r + \gamma v_{*}(s'))
            $$
    - Q:
            $$
            q_{\pi*}(s) = \sum_{r} \sum_{s'} P(s', r|s, a)(r + \gamma max_{a'} q_{\pi *}(s', a'))
            $$

