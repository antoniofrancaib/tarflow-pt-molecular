## Shape & Explore: Energy-Based Models

Having covered iterative dynamics, we now consider models that define unnormalized energies and rely on sampling dynamics to explore the distribution.

### Core Idea
Specify
\[
p_\theta(x) \propto e^{-E_\theta(x)}
\]
and sample with MCMC (Langevin, HMC). The sampling dynamics are the transport mechanism.

### Strengths
- Flexible densities and compositionality via additive energies
- Natural fit for scientific posteriors and constraints

### Trade-offs
- Mixing and convergence are challenging in high dimensions
- Training stability; evaluation without tractable normalizers

### When To Use
- Unnormalized targets; scientific inference where energies are meaningful
- As a principled prior or regularizer combined with learned transports

### Modern Links
- Denoising/score models connect to EBMs via score matching \([Vincent, 2011](https://arxiv.org/abs/1101.1152)) and \(p\)-energy formulations
- Classifier-as-energy views (e.g., JEM) \([Grathwohl et al., 2019](https://arxiv.org/abs/1912.03263))

Continue: [Hybrids & Conversions](./07-hybrids-and-conversions.md)


