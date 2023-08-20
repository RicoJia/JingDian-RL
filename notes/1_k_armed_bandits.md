## k-Armed Bandits

K-armed bandit is a slot machine has its own lever, and is called a “bandit” because it can empty your wallet like a thief :)

- Each bandit has a distribution of its reward
- We have a fixed number of levers to pull in action
So, you need to constantly check exploration vs exploitation (epsilon-greedy) and evaluate $Q_k(a) = 1/k(r_1 + ... r_k)$, the long term total reward till $kth$ step, based on each step's action and state.
    - $k$ is the number of times the action is called
  - $Q_{k}(a) = Q_{k-1}(a) + 1/k(R-Q_{k-1}(a))$, so we can simply add reward onto the existing Q function. (Dynamic Programming)
- Vanilla epsilon-greedy algorithm does not decay its choice
- There's also UCB methods: $Q_{new} = Q + \sqrt{\frac{ln(step)}{k}}$, so we are adding a term, that promotes the value of actions that haven't been explored yet.  

### Stochastic Gradient Ascend Method
- Boltzmann Distribution: the distribution of ideal gas speed, under constant temperature T. It can be roughly written as 
$$
P(E) = \frac{\omega_1 e^{-E_1}}{\omega_1 e^{-E_1} + \omega_1 e^{-E_2}...}
$$
- $E_n$ is directly related to the velocity
- [Good Explaination](https://stanford.edu/~ashlearn/RLForFinanceBook/MultiArmedBandits.pdf)
    - We have policy $\pi(a_k) = \frac{ e^{h_k}}{e^{h_1} + e^{h_2}...}$, where $h$ is a preference score
    - Now, we want to maximize score $E(R_t)=\sum_k \pi(a_k)r_k$
    - Gradient: $\frac{\partial E}{\partial h} = \sum \frac{\partial \sum_k \pi(a_k)r_k}{\partial h}$
        - $r_k$ has nothing to do with $h$. So, $\sum_k r_k \frac{\partial \pi(a_k)}{\partial \pi(h)}$
        - based on $pi$ definition, this is finally $E[r \dot (1/0 - \pi(a_k))]$, $1/0$ is one or zero
    - To update, $h_{k+1} = s_k + \alpha (r_t-\bar r_t)(1 - \pi(a))$, where we approximate the gradient as $(1 - \pi(a))$
        - We introduced average total reward as baseline $\bar r_t$, which helps reduce variance in $r$.
