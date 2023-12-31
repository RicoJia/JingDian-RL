# Model Free Prediction

why is it called tabular methods? In a small state space, we can write policy and value as tables. But in a large state space, we might need to approximate the value functions. Those functions may not converge, or converge to the right value

Here Prediction is to evaluate $V^{\pi}(s)$, the value of states under policy $\pi$

## Monte Carlo Methods

Many times, we don't know the model of the world $P(s',r|s,a)$. So, for a given $(s,a)$ we take average of next state, and rewards.

For example, we have the grid world. We want to have
    - $v(s)$, which could be all zeros, and associated rewards, and transitions.
    - policy $\pi(s)$ for each state

At this point, you don't know anything about the grid world. So, you do a bunch of experiments, in episode, **following $\pi$**. This way, you can estimate $V(s)$, by **law of large number**, it will converge to the true value of $V(s)$ over time (value is mean return). In other words, you use **samples** to approximate the expectation

### First-Visit vs Every-Visit  MC Methods

You do a bunch of episodes. In Each episode, you have $s_1, a_1, r_1 ...$ till T. and Total Reward $G=0$.
Then, you go back in time, from time $T$ to $0$: Say now you are at state $s$,

- First Visit: At the end of the kth episode, for the **first visit** of state $s$,
    1. $G_t(s)=r_t+yr_{t+1}...$, Total G: $G_{total}(s) += G_t(s)$
    1. $N(s)+=1$
    1. Then, $V_k(s) = G_t(s)/N(s)$

- Every Visit: At the end of each episode, for each visit of state $s$,
    1. $G_t(s)=r_t+yr_{t+1}...$, Total G: $G_{total}(s) += G_t(s)$
    1. $N(s)+=1$ (number of VISITS of the **state**)
    1. Then, $V_k(s) = G_{total}(s)/N(s)$

- Incremental: still need to wait for an episode to finish. But you update
    $$V_t(s) = V_{t-1}(s) + \alpha (G_{t}(s) - V_{t-1}(s))$$
  - $\alpha = 1/N(s)$ will be identical to every-visit
  - $\alpha > 1/N(s)$ will favour more recent updates, which applies to "non-stationary domains". E.g., machine wears out over time. 

- OPTIONAL: if you want to update policy $pi$:
    1. calculate $v(s) = s(s)/n(s)$, or $q(s,a) = s(s,a)/n(s,a)$
    1. you need optimal $(s,a, s')$ lookup
    1. $a_{policy} = \pi(a|s)$, get $s'=lookup(s,a_policy)$ if $v_(s')< v_(s)$, then $\pi(a|s)=a$

### Notes

- Bias:
  - Bias is $E(\hat{V} - V)$, the estimated V minus its true value. First-visit updates its V in iid fashion, so there's no bias. But in every-visit, there could be a bias because visits could be correlated.

- Variance and MSE:
  - Every-visit has smaller variance, because you get a lot more data, which are more correlated. That contributes to smaller $MSE = Var(V) + Bias(\hat{V})^2$

- MC only applies to Episodic MDP: average returns over the episode.  MC doesn't apply to something that goes on forever. One example is a loop with no way out.

## Temporal Difference Learning (TD)

In Monte Carlo method, we udpate using the mean value of the nth episode.
But in TD Learning, we can simplify this process by updating with the immediate reward, **which is online**
    $$v_{n+1}(s) = v_{n}(s) + \alpha (r + \gamma v_{n}(s') - v_{n}(s))$$. **This is really blending Monte Carlo into dynamic programming**, where bootstrapping is the DP part.

**Barto said, TD is the most central and novel to RL**. Intuition: use current esimtate instead of waiting for G at the end of episode
    - **GAMMA** can be 1 for episodes, but not for TD

### One step TD $TD(0)$

1. Initialize V(s), have a given policy
1. Loop for each episode until the terminal condition is reached:
    1. Take an action given by policy, observe next state $s'$, and its $v(s')$, $r$
    1. update Value
        $$v_{n+1}(s) = v_{n}(s) + \alpha (r + \gamma v_{n}(s') - v_{n}(s))$$
    1. update $s -> s'$

#### Some concepts

- TD Target is $R + \gamma v_{n+1}(s)$, to estimate total discounted reward $G_n(s)$.
- TD error is $R + \gamma v_{n+1}(s) - v_n(s)$
- **it's called bootstraping**, because you learn from another estimate. So, states are "resampled" MC is not considered using bootstrapping, because the inputs are randomly sampled. In other words, you reuse ```(s, a, r, s')```

#### Theoretical Comparisons

- TD to incremental MC:
    - When you have episode, and $\alpha = 1/N$, TD is incremental MC
- TD to DP: 
    - TD error doesn't have to estimate $\sum_{s'} P(s'|s)V(S')$. Because you don't know the transition model.
    - if $\alpha = 1/N$, TD effective estimates $V(s)$ with average, which is effectively $r + \sum_{s'} P(s'|s)V(S')$

### Characteristics and Comparisons With Other Methods

| Characteristic | DP | MC | TD |
|:------------|:------------|:--------------:|--------------:|
| Model Free | X | V | V |
| Continuouing domain | V | X | V |
| Bootstrap | V | X | V |
| Must be Markovian | V | X | V |
| Consistency | V | V | V |
| Bias | X | V&X | V |
| Variance | N/A | High | Mid |


Some Explanations

- Continuouing domain = non episodic. DP is non episodic, because you don't need clear episode to finish updating
- **Markovian: monte carlo doesn't require markovian transition model. TD and DP however do**, because their boostrapping uses previous value estimate in the same episode (resampling), which assumes that's sufficient enough for representing the past history.
- Consistency: means at infinite time will finally converge to the correct value. They all do.
    - But TD(0) with function approximation may not converge
- Bias: means in finite horizon, the expected value function values are off from the true values. DP and TD will introduce a bias, because each update to the value uses bootstraping, and might be correlated with previous value, which could be off. But First-time MC updates are independent from each other, so that's unbiased.
- Variance: MC is high per update, because you have different samples coming in.TD, not as much
- Data efficiency: MC is higher, because all updates within the same episode will kick in in the same update. Image an episode finally yields a high reward. When it first occurs, for TD $(\lambda)$, only $\lambda$ steps ahead can immediately see it. You potentially need the same episode to happen $L-\lambda$ times ($L$ is the episode's length)for the starting state to see it.

### In Batches (offline)

You have N epiodes, you keep replaying them, batch by batch, and update the value only at the end of a batch, until convergence. **TODO: I think we do $r = mean(r_episode)$?**

E.g., (Barto example 6.4), You are provided with 8 episodes of observations:

- 6 episodes of (B, 1)
- 1 epsode (B,0)
- 1 epsode (A, B, 0)

In Batch MC:
- V(B) = 6/8 = 3/4
- V(A) = 0/1 = 0 (you have only 1 episode with it)

In TD, say $y=1, \alpha=1/N$:
- V(B) = $(1-\alpha) V(B) + \alpha(r + V(terminal))$. $r(B) = 3/4$ (because we're looking at average, Right?). Then, you can easily get $V(B) = 3/4$
- V(A) = $(1-\alpha) V(A) + \alpha(r + V(B))$. Since $V(B)= 3/4$, here $V(A) = 3/4$

Difference is: 
- MC converges to values min MSE w.r.t the batch
- TD converges to Maximum Likelihood MDP, you are estimating by counting and averaging: $\hat{P}(s'|s,a) = 1/N*\sum_k\sum_t 1$ for each transition, and $\hat{r}=1/N * \sum_k \sum_t r$

### Certainty Equivalence Model
In Economics, certainty equivalent is "how much guaranteed benefit" you'd rather receive, than taking a certain risk for a certain higher outcome. E.g, if you choose $40 over gambling for $100 with 50% winning probability, $40 is your certainty equivalent. 
In control, CE model is a model where you can assume the controller can perform under all uncertainties. The above TD model is one with that assumption.
## Importance Sampling
Motivation: Sometimes, it's very difficult to get new data. If you have old data but under a different policy, you can use it.

Math: your target distribution is $P(X)$, but your old distribution is $Q(X)$. So expectation of $f(x)$ where $x$ is realization:
$$
E_{P}(f(X)) = \int_{x} p(x)f(x) dx 
 \\
 = \int_{x} q(x) \frac{p(x)}{q(x)} f(x) dx
 = E_{Q}(\frac{p(x)}{q(x)} f(x))
 \\
 \approx \sum_{i} \frac{p(x_i)}{q(x_i)} f(x_i)
$$

Implications:
- let $Y = \frac{p(x)}{q(x)} f(x)$
- If $P(X)/Q(X)$ is too high in a region, then in that region, $Var(X) = E[Y^2]-E[Y]^2$ could be high 

For estimating values: say we have behavior policy $\pi^b$, and target policy $\pi^t$. We collect m epsiodes, each episode can be represented as a trajectory: $h_m = (s_0, a_0, r_0, s_1, a_1, r_1...)$. Then, we can get $G(h_i) = r_0 + yr_1 ...$
$$
V_{\pi^t}(s) = 1/m \sum_i^m{G_i(s)}
\\
= 1/m \sum_i^m{ \frac{p(h_i | \pi^t, s)}{p(h_i | \pi^b, s)} G(h_i)}
\\
= 1/m \sum_i^m{ \frac{\prod \pi^t(a_i|s_i)}{\prod \pi^b(a_i|s_i)} G(h_i)}
$$

Then $G_i(s)$


Additionally: <https://towardsdatascience.com/importance-sampling-introduction-e76b2c32e744>