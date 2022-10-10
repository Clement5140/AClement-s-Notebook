# On the Convergence Theory of Gradient-Based Model-Agnostic Meta-Learning Algorithms

文章研究了MAML和FO-MAML的收敛理论，根据梯度范数分析了它们对于非凸损失函数的计算复杂度和能够达到的准确率级别(level)。

作者提出MAML对于任意的\\(\epsilon\\)，可以在\\(O(1/\epsilon^2)\\)次迭代内找到一个\\(\epsilon\\)-一阶导不动点，每次迭代复杂度为\\(O(d^2)\\)，\\(d\\)为问题维数，而FO-MAML将每次迭代的复杂度降低为了\\(O(d)\\)，但不能保证达到想要的准确率级别。

最后作者提出Hessian-free MAML(HF-MAML)算法，既能保有MAML的所有理论保证，又能降低每次迭代的复杂度为\\(O(d)\\)。

## MAML算法和FO-MAML算法

作者用自己的符号表示了两个算法：

![1](assets/HF-MAML-1.png)

![2](assets/HF-MAML-2.png)

其中MAML算法需要求损失函数的二阶导Hessian矩阵，而FO-MAML将其近似掉了，作者这里设置计算二阶导的数据集\\(D_h^i\\)和计算一阶导的数据集\\(D_o^i\\)相互独立，为了使用更小的\\(D_h^i\\)来减少计算量。

## Hessian-free MAML

对于任意函数\\(\phi\\)，其Hessian矩阵和任意向量\\(v\\)的乘积可以近似为

$$
\nabla^2\phi(w)v \approx \Bigg[\frac{\nabla\phi(w + \delta v) - \nabla\phi(w - \delta v)}{2\delta}\Bigg]
$$

其误差不超过\\(\rho\delta||v||^2\\)，\\(\rho\\)为\\(\phi\\)的Hessian矩阵的利普希茨连续常数。

将其带入之前MAML的更新中得到

$$
d_k^i := \frac{\tilde{\nabla}f_i\Big(w_k+\delta_k^i\tilde{\nabla}f_i(w_k-\alpha \tilde{\nabla}f_i(w_k,D_{in}^i),D_o^i),D_h^i\Big)-f_i\Big(w_k-\delta_k^i\tilde{\nabla}f_i(w_k-\alpha \tilde{\nabla}f_i(w_k,D_{in}^i),D_o^i),D_h^i\Big)}{2\delta_k^i}
$$

$$
w_{k+1} = w_k - \beta_k \frac{1}{B} \sum_{i \in B_k} \Big[\tilde{\nabla}f_i(w_k-\alpha \tilde{\nabla}f_i(w_k, D_in^i), D_o^i) - \alpha d_k^i\Big]
$$

HF-MAML的算法如下：

![3](assets/HF-MAML-3.png)

## 理论分析
