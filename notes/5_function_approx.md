# Value Function Approximation

If our state space is too big, we simply can't do MC/TD to learn the value function. So, to represent value function $V(s,w)$, we can use: 
    - Linear combo of features (Differentiable)
    - decision tree (e.g., learn a medical treatment)
    - Deep Neural network (also differentiable)
    - Nearest neighbour, or even, fourier wavelet bases

### SGD
Say hypothetically, we have loss $J(w) = E[(V^{\pi}(s)^-\hat{V}(s;w))^2]$. To reach local minima of J, since the above is min-squared-error we define w update 
$$
\triangle w = -1/2 a \nabla_w(J(w))
$$
Where, 
$$
\nabla_w(J(w)) = E[2(V^{\pi}(s)-\hat{V}(s;w))]*(-\nabla_v(V(s;w)))
$$
So
$$
\triangle w = a E[(V^{\pi}(s)-\hat{V}(s;w))]*(\nabla_v(V(s;w)))
$$
People would like to do SGD in minimatches.

### Feature Vector:
a feature vector may not be markovian (a.k.a, partial aliasing). E.g, 180 deg range finder, your feature vector is $[dist(0), dist(1), ...]$ the hallway on first and second floor appear the same. So they are not markovian?

$$
V(s;w) = X(s)^TW
$$

So 
$$
\triangle w = a E((V^{\pi}(s)- X(s)^TW)) * X^T
$$