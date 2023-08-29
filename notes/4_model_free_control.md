# Model Free Control

What does control mean: We want to identify a policy with high expected rewards. Might take many steps to evaluate how good an earlier decision is. 
    - Here updating $Q(s,a)$ will be more handy in finding the best policy
    - Setting $Q(s,a)$ everywhere can be beneficial to exploration

## Q evaluation

1. Collect N episodes $[(s,a,r)...]$
2. for each $(s,a,r)$, $G = r_0 + yr_1 ...$
3. for every time / first time, Get corresponding $G_{total}(s,a)$
4. $Q(s,a) = G_{total}(s,a)/N(s,a)$, $N+=1$

## Policy Evaluation with Exploration
Epsilon soft: with $\epsilon$ probability that you uniformly select an action ($\epsilon/|A|$ for each action).

Theorem: monotonic increase of $V_{\pi_{i+1}} > V_{\pi_{i}}$. Assuming V are computed EXACTLY. I.e., not when we are just estimating.
Proof:
1. We know $V_{\pi_{i+1}}(s) = max_a Q(s, a) >= Q_{\pi_i}(s, \pi_{i+1}(s))$, TODO?
2. 
$$
Q_{\pi_i}(s, \pi_{i+1}(s)) = \sum_a \pi(a|s)Q_{\pi_i}(s,a)
\\
= \epsilon/A (\sum_a Q(s,a)) + (1-\epsilon) max_a Q_{\pi_i}(s,a)
= ... + (\sum_a [\pi(a|s) - \epsilon/A])max_a Q_{\pi_i}(s,a)
\\
\ge ... + (\sum_a [\pi(a|s) - \epsilon/A]*Q_{\pi_i}(s,a))
= \sum_a \pi(a|s) Q_{\pi}(s,a) = V_{\pi}
$$

### GLIE 
Greedy in the Limit of Inifite Exploration: $\epsilon = 1/i$ to reduce unwanted exploration
    - Definition: as all $(s,a)$ pairs are visited to infinite number of times, policy should converge to greedy policy.
    - 

### Monte Carlo Online Control

1. Do policy Evaluation above.
2. At the end of an episode, do policy improvement: $\pi_{i+1} = \epsilon(Q_{i+1})$

## On, Off Policy
**One problem is you can't gurantee that all states (and maybe actions) are explored**
So, you can choose to:
1. choose random start states for each episode, following some policy
2. Have a stochastic policy of action for each state. **Note that in the above example, you don't change policy.**
    - the policy for selecting actions is called "behavior policy", policy for updating Q function is called "update policy"
    - **on-policy** is to keep updating the policy $\pi (a|s)$, i.e., the behavior policy and the update policy are the same
    - **off policy** has different behavior and update policies
### On-policy, SARSA
1. set policy randomly, $q(s,a)$ to zeros
2. Take action $a_0$, get $(r_0, s_1)$
3. Loop:
    1. Take action $a_{t+1}$, get $(r_{t+1}, s_{t+2})$). So now you have: 
        $$
        S_t, a_t, r_t, S_{t+1}, a_{t+1}
        $$
    1. Update Q: 
        $$
        Q(s_t, a_t) = \alpha Q(s_t, a_t) + (1-\alpha)(r_t + yQ(s_{t+1}, a_{t+1}))
        $$
    1. Policy Improvement:
        $$
        \pi(s_t) = argmax Q(s_t, a_t) | \epsilon
        $$
    1. t += 1
1. on policy, similar to the above mc method. but different in that
    - It's on the go! No need to wait for episode to finish
    - we evaluate realization of $q(s,a)$, instead of finding the max
1. Here, we can set **the learning weight** $\alpha_t = 1/t$. But usually, we want to incorporate domain info in it.
1. Could be better than Q learning in early stages, especially when negative reward is a lot.

### Off Policy Q Learning
1. Instead of $Q(s_t, a_t) = \alpha Q(s_t, a_t) + (1-\alpha)(r_t + yQ(s_{t+1}, a_{t+1}))$, we do $Q(s_t, a_t) = \alpha Q(s_t, a_t) + (1-\alpha)(r_t + y*max_a' Q(s_{t+1}, a'))$
    - No matter how to initialize, it will converge
    - doesn't back propagate the learning? So could be slower than MC
 
### Maximization Bias
**Q(s'a') is seen as a random variable.**, then in Q Learning, because you always choose action that lead to current $max Q(s', a')$, there's always a difference between the $E[max Q(s',a')]$, and the $max EQ(s',a')$. This difference is called bias, and we can show the bias is positive:

1. Math:
    - convex function $af(x)+(1-a)f(y) \geq f(ax + (1-a)y)$
        - $max(m,n)$ is convex. Proof:
            $$
            max(am_1 + (1-a)m_2, an_1 + (1-a)n_2) \leq max(am_1, an_1) + (1-a)max(m_2, n_2)
            $$
    - Jensen's inequality: for convex function $f(x)$, $f(E(x)) \leq E(f(x))$.
2. So even if each individual Q(s,a) is an unbiased estimate, a realization of it has noise. So in finite sampling, an estimate of V can be higher than it actually is: $V_{\hat{\pi}}(s) = max_a Q(s,a)$ is biased:
    $$
    V_{\pi} = max_a(E(Q(s,a))) \leq E(max_a(Q(s,a))) = V_{\hat{\pi}}(s)
    $$
    - Same thing applies to Q learning, where we overestimate the value of Q(s,a) by using argmax.
3. To mitigate the bias of Q, we are using double Q learning
    1. Instead of updating using one single Q function, and taking argmax, we have 2 Q functions:
        1. select $a_t$ based on $\epsilon$ greedy policy: $argmax_a [Q1(s,a)+Q2(s,a)]$
        - with 50% chance, update $Q1(s,a)=Q1(s,a) + \alpha[R + y Q2(s,a) - Q1(s,a)]$
    2. [Implementation](https://rubikscode.net/2021/07/20/introduction-to-double-q-learning/): Have two $Q$ tables: $Q1(s,a)$, $Q2(s,a)$.
F 


TODO:
1. Cliff walking (lots of negative rewards)
1. 