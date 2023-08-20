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

## Common Problems and Elements

- 4 main elements:
    - a policy (a look up table, but could be a searched function, in a complicated env)
    - a reward signal
    - a value function
    - and optionally, a model of the environment
- Common Problems:
    - Exploration vs exploitation: when to do the most optimal action the agent thinks to maximize profit, or do non-optimal things to explore actions it hasn't seen yet (so, risk involved)
    - Delayed reward: acting greedily could result in less future reward
- Some terms:
    - Model free: we have no idea of how the environment change. E.g., In tic-tac-toe, RL agent does not know what its opponent will do.
    - Bootstrap: you can pull yourself up by pulling your own bootstrap. It's the first piece of code being run on a machine.

### Applications
In Advertisement, to maximize total revenue, we want to show viewers ads that they will most likely like. This is called "impression allocation", where an impression is an ad pop-up. Platforms use collaborative filtering or content based filtering to rank sellers using historical scores, which considers how many customers the sellers likely to get. The current algorithms consider the sellers' similarities to customers, but not discounts sellers make to attract more customers. So, reinforcement learning can be used to better evaluate seller's behavior. Also, reinforcement learning can be used to detect fradulent behaviors on a platform?

On the other hand, sellers can use RL to allocate advertising budget across platforms. There's online bidding platforms that show a seller's ad if they win the bid. A seller can automatically bid based on the total reward of the combo of platforms they use.

#### Research - Industry:
- Google
    - i-Sim2Real: Reinforcement Learning of Robotic Policies in Tight Human-Robot Interaction Loops 
    - GoalsEye: Learning High Speed Precision Table Tennis on a Physical Robot
    - DRL: can control 19 coils, 
    - DeepNash
    - Deepmind: sparrow model
    - Deepmind: in-context learning
    - DeepMind在PNAS发文，打开AlphaZero 
- Stanford Feifei Li:
    - Free an intelligent agent from books, and to learn in the society
- Misc
    - FinRL: automated transactions
    - AutoRL for RNA and go chess
    - PID tuner using DRL

- 
### Resources
- Stanford CS234: 
    - [Course link](https://www.youtube.com/watch?v=FgzM3zpZ55o&list=PL-myaKI4DslUer7Pwkamk92F4PAFyBTPW&index=1)
    - [Schedule:](https://web.stanford.edu/class/cs234/)
    - Solution: https://github.com/tallamjr/stanford-cs234/tree/master/slides
    - Sample implementation https://github.com/zlpure/CS234/blob/master/assignment1/vi_and_pi.py

- Review this [catalogue after the CS234 course:](https://github.com/kmario23/deep-learning-drizzle#loudspeaker-probabilistic-graphical-models-sparkles)
- Optional 
    - 动手写RL, 教程。https://datamachines.xyz/the-hands-on-reinforcement-learning-course-page/ 
    - 中文： https://hrl.boyuai.com/chapter/1/%E5%88%9D%E6%8E%A2%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0

- Why useless
    - Still a lot of $$ flows to game
    - Robot Folding clothes, Opening Door, still seems to be far??
        - Continuous action space: 10^9. More complicated than RL applied in a game. Which has clear actions.
        - There's RL mechanism that solves door (2021)
        - 80% of AI research output is to perform one specific task.
        - Tuning, and tuning, until it solves more instances of this problem
    - AI is probably over-hyped
        - Production Reinforcement learning models?
        - does incremental learning and “lifelong learning”?
#### Reinforcement Learning Self Learning Schedule
1. July 27 - 29: 
    - Finite MDP Lecture, https://github.com/tallamjr/stanford-cs234/blob/master/slides/lecture02.pdf
        - Take notes
        - Homework
1. July 30 - 31: (Actually: July30 - Aug 5, 4 days late)
    - Frozen Lake Implementation 
    - Organize the MDP notes.
    - Lessons Learned:
        - FIFA game was a huge distraction. (1-2 days)
        - 1 day spent on debugging and solving Julia Import problems.
1. Aug 5-6:
    - Model Free methods, notes
1. Aug 13-15, 15-17:
    - Model Free Assignments