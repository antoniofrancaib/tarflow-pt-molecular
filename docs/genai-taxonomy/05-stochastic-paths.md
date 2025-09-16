## Evolve: Stochastic Paths

From one-shot maps, we move to iterative dynamics that evolve uncertainty into data through time.

### Core Idea
Move probability mass along a time axis with SDE/ODE dynamics, parameterized by scores, drifts, or velocities.

### Families
- **Score-based diffusion**: Learn the score \(\nabla_x \log p_t(x)\) and simulate reverse-time SDE or an equivalent ODE.
  - DDPM \([Ho et al., 2020](https://arxiv.org/abs/2006.11239))
  - Score-based SDEs \([Song et al., 2020](https://arxiv.org/abs/2011.13456))
  - DPM-Solver \([Lu et al., 2022](https://arxiv.org/abs/2206.00927))

- **Schrödinger Bridges**: Bridge between prior and data marginals via entropic optimal transport.
  - Diffusion Schrödinger Bridge \([De Bortoli et al., 2021](https://arxiv.org/abs/2106.01357))

- **Flow Matching / Rectified Flows**: Learn vector fields to integrate ODEs from prior to data.
  - Flow Matching \([Lipman et al., 2022](https://arxiv.org/abs/2210.02747))
  - Rectified Flow \([Liu et al., 2022](https://arxiv.org/abs/2209.03003))

### Strengths
- State-of-the-art fidelity on many modalities
- Stable training and flexible conditioning hooks

### Trade-offs
- Iterative sampling cost (time–quality trade-off)
- Likelihood estimation may be costly or indirect (unless ODE likelihoods are used)

### Speedups
- Distillation to fewer steps (consistency, progressive distillation)
- Better numerical solvers; caching and reuse of intermediate features

### Pitfalls
- Aggressive guidance harms diversity; solver mis-specification can introduce artifacts

Continue: [Energy-Based Models](./06-energy-based-models.md)


