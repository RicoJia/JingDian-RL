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
Eg: 2 fair coins, flip both of them. If you get a head, +1, get a tail, -1. After 1 flip, tell me: 1. which coin is better; 2. Expectation of the coin flip
1. Say you get head from coin 1. So coin 1 is better; expectation is 1
2. What's the expected value of the answer to question 2, **given coin1 is the best**? 3 scenarios of getting 1, 1 scenario of getting 0.
    so 3/4 * 1 - 1/4 * 1 = 0.5. But the real expectation of flipping coin1 is 0.
    **A better way to interpret this is**: if you want the expectation of the better coin, don't use the old estimate, because you have 3/4 chance that you got a 1 - so most likely you have a one. This is also **survivorship bias.**
3. So that's called maximization bias. To mitigate that, flip coin 1 again. At least that's a realization of the true probability.

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
    - Same thing applies to Q learning, where we 
        1. first pick the "better action" (like choosing the better coin)
        2. overestimate the value of Q(s,a) by using argmax.

3. To mitigate the bias of Q, we are using double Q learning
    1. Instead of updating using one single Q function, and taking argmax, we have 2 Q functions:
        1. select $a_t$ based on $\epsilon$ greedy policy: $argmax_a [Q1(s,a)+Q2(s,a)]$
        - with 50% chance, update $Q1(s,a)=Q1(s,a) + \alpha[R + y Q2(s,a) - Q1(s,a)]$
        1. This way, we don't let the action selected from $\pi$ give us the **survivorship bias**. For updating Q, we ask to flip the coin again, not considering the survived action.
    2. [Implementation](https://rubikscode.net/2021/07/20/introduction-to-double-q-learning/): Have two $Q$ tables: $Q1(s,a)$, $Q2(s,a)$.

## Bonus: Expected SARSA
Instead of using a realization of the next $Q(s_{t+1}, a_{t+1})$, we use its exected value: 
    $$
    Q(s_t, a_t) = Q(s_t, a_t) + \alpha (r + y \sum_{A} \pi(a|s_{t+1})Q(s_{t+1}, a) - Q(s_t, a_t) )
    $$
    - Biased (like Q learning), but should converge
    - Reduced variance? Because you are considering all actions
    - Sure, more computation. Computation complexity vs sample efficiency

TODO:
1. Cliff walking (lots of negative rewards)
1. 