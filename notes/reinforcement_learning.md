

### On-policy

on policy, similar to the above mc method. but different in that
    - we evaluate q(s,a) on the go
    - update policy **Say we do it first time, with Q(s,a)**

1. Iteration
    - $q(s,a)$, which could be all zeros
    - Returns $s(s,a)$ for each state and action, which should be all zeros
    - number of visits at each state $n(s,a)$ where every entry is zero
    - policy $\pi$, epsilon-soft
    At this point, you don't know anything about the grid world. So, you do a bunch of experiments, in episode.
1. In one episode, you have $s_1, a_1, r_1 ...$ till T. and Total Reward $G=0$. ,
1. Then, you go back in time, from time $T$ to $0$: Say now you are at state $s$,
    1. $G_t=\gamma G+r_t$,
    1. **If $s_t$ has not shown up yet:**
        1. $n(s,a)+=1$
        1. $q(s,a)+=G_t/n(s,a)$.
        1. get set of optimal actions  $A^*=argmax_a Q(s,a)$
        1. update policy of actions $a$ in $s_t$, $A(s_t)$:
            - Non-optimal actions $\epsilon / |A(s_t)|$
            - optimal actions: $1 - \epsilon + \epsilon/|A(s_t)|$

1. When you are done, for each state, you have $q(s,a)$, and policy $\pi(s,a)$

### Importance Sampling

If you want to get expectation, variance, etc. from a distribution that's hard to sample from, consider sampling from an easier distribution.

- But you need ratio between the two distributions, at each value.

$$E(X) = \int xP(x)dx = \int x \frac{P(x)}{Q(x)}Q(x)$$

- people usually do $E(f(x)) = \int f(x)p(x)dx$

- However, if $Q(x)$ and $xP(x)$ distribution look too different, then $xP(x)/Q(x)$ is large at every $xP(x)$,
so $var = E[X^2]-E[X]^2$ will become larger in those regions?

### Importance Sampling

- Need example: <https://towardsdatascience.com/importance-sampling-introduction-e76b2c32e744>

1. TODO: notes are in IPAD

### Off Policy

Have a behavior policy $b(a_i|s_i)$, and a target policy $\pi (a_i|s_i)$. Use the behavior policy for generating new episodes, but
target policy to update and return.

- Video: <https://www.youtube.com/watch?v=bpUszPiWM7o>

1. TODO: notes are in IPAD

## TD Learning

1. on and off policy
    - SARSA updates its $Q$ using the actual action taken by $s'$, which corresponds to the behavior policy. So,
        it's **on-policy**
    - Q Learning updates its $Q$ always chooses the most rewarding action of $s'$, which could be different from its actual actions.
        So its update policy is different than its behavior policy, so it's **off-policy**
    - Expected Policy can be either on or off policy. Update policy you use $\pi(a|s)$, could be, or different than your action policy
    $b(a|s)$

### SARSA

It's basically the above, just swap them with $Q(s,a)$

1. Initialize $Q(s,a)$, have a given policy $\pi$
1. Loop for each episode until the terminal condition is reached:
    1. Take an action $a$ based on epsilon-greedy policy (greedy means "choosing the most rewarding policy")
    1. observe next state $s'$, $r$
    1. Take action $a'$ based on policy, update its $Q(s',a')$ (From the sequence of things to update, S, A, R, S, A)
    1. Update Q value
        $$Q_{n+1}(s, a) = Q_{n}(s, a) + \alpha (r + \gamma Q_{n}(s', a') - Q_{n}(s, a))$$
    1. update $s -> s'$

### Q Learning (watkins)
TD to Q learning: TD has a fixed policy, while Q learning always gets greedy policy

Compared to SARSA, instead of using the policy, We take optimal action when updating Q. But we are still using the policy to find the next action

1. Initialize $Q(s,a)$ arbitrarily, have a given policy $\pi$
1. Loop for each episode until the terminal condition is reached:
    1, Take an action $a$ given by policy $\pi, observe next state $s'$, $r$
    1. Update Q value
        $$Q_{n+1}(s, a) = Q_{n}(s, a) + \alpha (r + \gamma max_{a'} Q_{n}(s', a') - Q_{n}(s, a))$$
    1. update $s -> s'$

### Expected SARSA

Instead of updating with next actual action in SARSA
    $$Q_{n+1}(s, a) = Q_{n}(s, a) + \alpha (r' + \gamma Q_{n}(s', a') - Q_{n}(s, a))$$
We update with the expectated Q:
    $$Q_{n+1}(s, a) = Q_{n}(s, a) + \alpha (r' + \gamma \sum_{a'} \pi(a'|s')Q_{n}(s', a') - Q_{n}(s, a))$$

**Expected SARSA eliminates variance from random action selection**. Behaves better than SARSA.

### Maximization Bias and Double Q Learning

**Q(s'a') is seen as a random variable.**, then in Q Learning, because you always choose action that lead to current $max Q(s', a')$, there's always a difference between
the $E[max Q(s',a')]$, and the $max EQ(s',a')$. This difference is called bias, and we can show the bias is positive:

1. $max(x)$ is a convex function. For convex function, there's Jensen's inequality, which states:
    $$E[f(x)] >= f(E[x])$$
1. So, $bias = E[max Q(s',a')] - max E[Q(s',a')] >= 0$
1. So, the expectation of the max among Q(s'a') is always larger than the max of the expectation among Q(s'a')

So, there's double Q learning: TODO

[Implementation](https://rubikscode.net/2021/07/20/introduction-to-double-q-learning/): Have two $Q$ tables: $Q1(s,a)$, $Q2(s,a)$.
For each episode:
    1. $Q_total =Q_A+Q_B$, find the max action.
    1. flip a coin. If updating $Q_A$:
        $$
        Q_{A,n}(s,a) = Q_{A,n}(s,a) + \alpha[R + \gamma Q_{B,n}(s_{n+1}, a) - Q_{A,n}(s,a)]
        $$
        - switch QA, QB if we want to update QB.
    1. $s_n\rightarrow s_{n+1}$

- TODO:
  - How does this minimize bias?
  - why is SARSA online, QLearning offline?

### N step Bootstrapping

You basically use actions and updates to update $n$ steps afterwards.
$$
G_{t:t+n} = \sum_{m=0}^{m=n-1}[\gamma^{m} * R_{t+m}] + \gamma^(n) v_{k}(s_{t+n})
v_{k+1}(s_{n}) = v_{k}(s_{n}) + \alpha (G_{t:t+n} - v_{k}(s_{t+n})
$$

- If you haven't got to finish the first $n$ steps yet, just stick to the policy, don't do anything.

- for off-policy learning, you want to consider importance sampling

    -

### Examples and More

- [Grid World](https://towardsdatascience.com/introduction-to-reinforcement-learning-rl-part-3-finite-markov-decision-processes-51e1f8d3ddb7)
- Inverted Pendulum (Q learning)

- temporal difference learning: to update the value function, we have a small feedback loop with the future value:
  - We first play the game
  - Then we back up after we finish, update $V(s) = V(s) + \alpha (V(s') - V(s))$,
    - wheren $\alpha$ is a step size parameter, positive integer
