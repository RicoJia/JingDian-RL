## Monte Carlo Control

1. Average Undiscounted Rewards across 10 runs, per episode:
![avg_plot](https://github.com/RicoJia/Omnid_Project/assets/106101331/69449482-c901-469c-8259-43936da487a9)
    - Test conditions:
        - Terminal reward is 50 (hitting the goal), -50 (getting into a hole); -0.1 for getting stuck, and -0.01 for taking a step
        - We update policy after each episode.
        - An exponential decay policy of $\epsilon$ is adopted, so the final policy will converge greedily. (GLIE)
    - A few things to note:
        - The curve settles at -4. This shows that after 2000 episodes, the algorithm converges. It's negative because of the penalties above.
3. Visualize an Episode:

    <div style="display: flex; justify-content: center;">
        <img src="https://github.com/RicoJia/JingDian-RL/assets/106101331/2d4412c2-e8f4-48d5-a609-fd0d7300c4e8" alt="image1" width="100"/>
        <img src="https://github.com/RicoJia/JingDian-RL/assets/106101331/70def7c1-cf3f-4406-b2b2-b7bc6574d503" alt="image2" width="100"/>
    </div>