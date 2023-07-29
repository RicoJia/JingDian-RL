## Survey
There are three types of machine learning cateogories:

- Supervised:
  - Learns from labelled data to predict the outcome of a given problem, by minimizing the prediction loss e.g., MNIST.
  - Iteratively evaluate by using SVM, Decision Tree, K-nearest neighbors, Deep Learning.
- Unsupervised:
  - Finds commonalities and structure between uncategorized data.
  - K-means, apriori, or PCA.
- Reinforcement Learning:
  - Goal is given an environment, action and state space, we want to optimize the total reward. So the goal is different from
    unsupervised learning.
  - Over time, we want to learn a series of optimal actions.
  - There are three types of reinforcement learning:
    - Model based, MDP
    - Model free:
      - Value based: We optimize value of state to predict, or state-action pair and control optimally
        - sarsa
        - q learning
        - When the state or state-action space is small enough, we can use a table to represent it (aka tabular learning). But when the state space is large, we want to use function approximation methods.
      - Policy based: (high variance during training?)
        - Reinforce (monte carlo policy gradient)
        - Deterministic Policy Gradient (DPG)
        - This is more similar to how we think: to figure out a strategy, unlike value based methods, learning the values of all states & actions.
      - Actor-critic: evaluate both the policy and the value with statble convergence.

Deep learning helps reinforcement learning to learn features much faster (TODO, ?), which  allows us to learn larger and more complex goals.
A reinforcement learning problem can be abstracted as an Markov Decision Process. But sure, this Markov Decision Process can be very large.

### Applications
In Advertisement, to maximize total revenue, we want to show viewers ads that they will most likely like. This is called "impression allocation", where an impression is an ad pop-up. Platforms use collaborative filtering or content based filtering to rank sellers using historical scores, which considers how many customers the sellers likely to get. The current algorithms consider the sellers' similarities to customers, but not discounts sellers make to attract more customers. So, reinforcement learning can be used to better evaluate seller's behavior. Also, reinforcement learning can be used to detect fradulent behaviors on a platform?

On the other hand, sellers can use RL to allocate advertising budget across platforms. There's online bidding platforms that show a seller's ad if they win the bid. A seller can automatically bid based on the total reward of the combo of platforms they use.