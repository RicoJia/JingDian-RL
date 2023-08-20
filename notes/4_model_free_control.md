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
Greedy in Limit of Inifite Exploration: $\epsilon = 1/i$ to reduce unwanted exploration

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