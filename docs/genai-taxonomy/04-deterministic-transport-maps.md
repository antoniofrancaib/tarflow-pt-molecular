## Transform: Deterministic Transport Maps

We now turn from sequential sampling to one-shot generation via learned transport maps.

### Core Idea
Learn a deterministic map \(T_\theta\) that pushes a simple base distribution (e.g., Gaussian) to data: \(x = T_\theta(z)\) with \(z \sim p_0\).

### Branches
- **Normalizing Flows / CNFs**: Invertible maps with tractable log-determinants enabling exact likelihood and fast sampling; ODE-based continuous flows (CNFs) enable flexible architectures.
  - RealNVP \([Dinh et al., 2016](https://arxiv.org/abs/1605.08803))
  - Glow \([Kingma & Dhariwal, 2018](https://arxiv.org/abs/1807.03039))
  - FFJORD (CNF) \([Grathwohl et al., 2018](https://arxiv.org/abs/1810.01367))

- **GANs**: Implicit generator maps trained via adversarial games; often top-tier fidelity and low latency, but no tractable likelihood and potential stability/mode coverage issues.
  - GAN \([Goodfellow et al., 2014](https://arxiv.org/abs/1406.2661))
  - BigGAN \([Brock et al., 2018](https://arxiv.org/abs/1809.11096))
  - StyleGAN2 \([Karras et al., 2019](https://arxiv.org/abs/1912.04958))

- **Consistency / One-step Models**: Distill multi-step generative processes (often diffusion) into near one-shot maps for speed.
  - Consistency Models \([Song et al., 2023](https://arxiv.org/abs/2303.01469))
  - Progressive Distillation \([Salimans & Ho, 2022](https://arxiv.org/abs/2202.00512))

### Use When
- You need low latency (GANs; distilled maps)
- You need exact likelihoods and fast sampling (flows)

### Pitfalls
- Support mismatch between base and data; mode coverage issues (especially in GANs)
- Computational cost from Jacobian log-determinants (flows)

Continue: [Stochastic Paths](./05-stochastic-paths.md)


