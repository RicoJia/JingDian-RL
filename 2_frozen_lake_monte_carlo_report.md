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
    ```
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
    (Down)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
    (Down)
    SFFF
    FHFH
    [41mF[0mFFH
    HFFG
    (Right)
    SFFF
    FHFH
    F[41mF[0mFH
    HFFG
    (Down)
    SFFF
    FHFH
    FFFH
    H[41mF[0mFG
    (Right)
    SFFF
    FHFH
    FFFH
    HF[41mF[0mG
    (Right)
    SFFF
    FHFH
    FFFH
    HFF[41mG[0m
    ```