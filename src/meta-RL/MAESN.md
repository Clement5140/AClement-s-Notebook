# Meta-Reinforcement Learning of Structured Exploration Strategies

Prior tasks can be used to inform how exploration in new tasks should be performed.

A meta-RL algorithm that adapts to new tasks by following the policy gradient, while also injecting learned **structured stochasticity** into a latent space to enable effective exploration.

Effective exploration strategies must select randomly from among the *potentially useful* behaviors, while avoiding behaviors that are highly unlikely to succeed. MAESN leverages this insight to acquire significantly better exploration strategies by incorporating learned **time-correlated noise** through its meta-learned latent space, and training both the policy parameters and the latent exploration space explicitly for fast adaptation.

## Preliminaries



## Model Agnostic Exploration with Structured Noise
