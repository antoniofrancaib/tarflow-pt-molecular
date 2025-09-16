## Factorize: Autoregressive Models

Following the sampling taxonomy, we first examine models that factorize the joint via the chain rule and sample sequentially.

### Core Idea and Signature Equation
Autoregressive (AR) models maximize exact likelihood by decomposing
$$
p_\theta(x) = \prod_{t=1}^T p_\theta(x_t\mid x_{<t}).
$$
Sampling is sequential: draw \(x_1\), then \(x_2\mid x_1\), and so on.

### Strengths
- **Exact likelihood** and calibrated perplexity/per-token NLL
- **Strong scaling** on discrete sequences (text/code)
- **Simple training** via teacher forcing

### Trade-offs
- **Slow sampling** due to sequential generation
- **Exposure bias** and drift in long rollouts
- **Long-range planning** often needs retrieval/tools or planners

### Where It Shines
- Large language and code models based on Transformers \([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762); [Brown et al., 2020](https://arxiv.org/abs/2005.14165))
- AR audio and image models (e.g., WaveNet; PixelCNN \([van den Oord et al., 2016](https://arxiv.org/abs/1601.06759)))

### Pitfalls
- Dataset contamination and memorization
- Length bias and context truncation effects
- Degradation with extremely long contexts without augmentation (RAG/tools)

Continue: [Deterministic Transport Maps](./04-deterministic-transport-maps.md)


