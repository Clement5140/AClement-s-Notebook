# Curiosity-driven Exploration by Self-supervised Prediction

In many real-world scenarios, rewards extrinsic to the agent are extremely sparse or missing altogether, and it is not possible to construct a shaped reward function.

In reinforcement learning, intrinsic motivation/rewards become critical whenever extrinsic rewards are sparse.

Two broad classes:

* encourage the agent to explore “novel” states
* encourage the agent to perform actions that reduce the error/uncertainty in the agent’s ability to predict the consequence of its own actions

The effectiveness of curiosity formulation in all three of these roles:

* solving tasks with sparse rewards
* helps an agent explore its environment in the quest for new knowledge
* learn skills that might be helpful in future scenarios

## Curiosity-Driven Exploration

Intrinsic curiosity reward generated by the agent at time t: \\(r_t^i\\), extrinsic reward: \\(r_t^e\\), sum: \\(r_t = r_t^i + r_t^e\\), with \\(r_t^e\\) mostly zero.

Policy: \\(\pi(s_t ; \theta_P)\\) by a deep neural network with parameters \\(\theta_P\\).

Given the agent in state \\(s_t\\), it executes the action \\(a_t \sim \pi(s_t ; \theta_P)\\).

\\(\theta_P\\) is optimized to maximize the expected sum of rewards:

$$
\max_{\theta_P} \mathbb{E}_{\pi(s_t ; \theta_P)} [\Sigma_t r_t]
$$

Asynchronous advantage actor critic policy gradient (A3C).

### Prediction error as curiosity reward

If not the raw observation space, then what is the right feature space for making predictions so that the prediction error provides a good measure of curiosity?

Divide all sources that can modify the agent’s observations into three cases:

* things that can be controlled by the agent;
* things that the agent cannot control but that can affect the agent (e.g. a vehicle driven by another agent), and
* things out of the agent’s control and not affecting the agent (e.g. moving leaves).

A good feature space for curiosity should model (1) and (2) and be unaffected by (3).

### Self-supervised prediction for exploration

A general mechanism for learning feature representations such that the prediction error in the learned feature space provides a good intrinsic reward signal.

Train a deep neural network with two sub-modules:

* the first submodule encodes the raw state \\((s_t)\\) into a feature vector \\(\phi(s_t)\\).
* the second submodule takes as inputs the feature encoding \\(\phi(s_t), \phi(s_{t+1})\\) of two consequent states and predicts the action \\((a_t)\\) taken by the agent to move from state \\(s_t\\) to \\(s_{t+1}\\).

Training this neural network amounts to learning function \\(g\\) defined as:

$$
\hat{a}_t = g(s_t, s_{t+1} ; \theta_I)
$$

The neural network parameters \\(\theta_I\\) are trained to optimize:

$$
\min_{\theta_I} L_I(\hat{a}_t, a_t)
$$
The learned function \\(g\\) is also known as the **inverse dynamics model** and the tuple \\((s_t, a_t, s_{t+1})\\) required to learn \\(g\\) is obtained while the agent interacts with the environment using its current policy \\(\pi(s)\\).

Train another neural network:

$$
\hat{\phi}(s_{t+1}) = f(\phi(s_t), a_t ; \theta_F)
$$

The neural network parameters \\(\theta_F\\) are optimized by minimizing the loss function \\(L_F\\):

$$
L_F(\phi(s_t), \hat{\phi}(s_{t+1})) = \frac{1}{2} \parallel \hat{\phi}(s_{t+1}) - \phi(s_{t+1}) \parallel _2^2
$$

The learned function \\(f\\) is also known as the **forward dynamics model**.

The intrinsic reward signal is computed as:

$$
r_t^i = \frac{\eta}{2} \parallel \hat{\phi}(s_{t+1}) - \phi(s_{t+1}) \parallel _2^2
$$

Intrinsic Curiosity Module (ICM).

The overall optimization problem:

$$
\min_{\theta_P, \theta_I, \theta_F} [-\lambda \mathbb{E}_{\pi(s_t;\theta_P)}[\Sigma_t r_t] + (1-\beta) L_I + \beta L_F], 0 \leq \beta \leq 1, \lambda > 0.
$$
