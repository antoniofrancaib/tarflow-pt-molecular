## Hybrids & Conversions Between Paradigms

With the four sampling lenses in hand, many practical systems combine or convert between them to capture the best of each world.

### Examples
- **Flow Matching / Rectified Flows**: Train ODE dynamics (evolve) and optionally distill to a one-shot map (transform) for speed \([Lipman et al., 2022](https://arxiv.org/abs/2210.02747); [Liu et al., 2022](https://arxiv.org/abs/2209.03003)).

- **VAEs with Flow Priors / Diffusion Decoders**: Latent-variable modeling with powerful priors or decoders drawn from transform/evolve families.
  - IAF \([Kingma et al., 2016](https://arxiv.org/abs/1606.04934))
  - LDMs \([Rombach et al., 2022](https://arxiv.org/abs/2112.10752))

- **Autoregressive + Latent**: Language models with VAE/flow priors; latent-AR for images (e.g., VQ-VAE-2 with AR priors).
  - VQ-VAE-2 \([Razavi et al., 2019](https://arxiv.org/abs/1906.00446))

- **EBM + Learned Transport**: Use a learned map to precondition MCMC for fast mixing, combining energy shaping with efficient exploration.

### Practical Notes
- Distillation offers large latency wins at modest fidelity loss if schedules and losses are tuned
- Hybridization often improves controllability and sample efficiency

Continue: [Evaluation & Diagnostics](./08-evaluation-and-diagnostics.md)


