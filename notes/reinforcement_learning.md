



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
