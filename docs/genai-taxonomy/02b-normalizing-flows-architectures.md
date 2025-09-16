### Three Normalizing Flow Architectures: Coupling, Autoregressive, Residual

This note presents three canonical invertible architectures for normalizing flows with emphasis on their invertibility, Jacobian structure, and log-determinant computation. The exposition follows the logic of the referenced presentation and standard surveys; see [video notes](https://youtubetodoc.s3.amazonaws.com/docs/youtube/DDq_pIfHqLs.md), [Papamakarios et al. (2021)](https://arxiv.org/abs/1912.02762), and [Kobyzev et al. (2020)](https://arxiv.org/abs/1908.09257).

#### Preliminaries

- Data $x \in \mathbb{R}^d$; base variable $z \in \mathbb{R}^d$, density $p_Z$.
- Invertible map $g: \mathbb{R}^d \to \mathbb{R}^d$, composition of layers $g = g_K \circ \cdots \circ g_1$.
- Change of variables for one layer $h' = g(h)$:

$$ \log p_{H'}(h') = \log p_H(h) - \log\big|\det J_g(h)\big|. $$

Efficient flows are those where both the inverse and $\log|\det J|$ are tractable.

### 1) Coupling Layers

Coupling splits the input into two parts and transforms one part conditioned on the other. Let a partition $h = (h_a, h_b)$ with $h_a \in \mathbb{R}^{d_a}, h_b \in \mathbb{R}^{d_b}$, $d_a + d_b = d$. Define

$$ s, t = \mathcal{N}(h_a), \quad h_b' = e^{s} \odot h_b + t, \quad h_a' = h_a. $$

Here $\mathcal{N}$ is an arbitrary neural network; $s, t$ have shape $d_b$; $\odot$ is elementwise product. The full output is $h' = (h_a', h_b')$.

- Inverse: $ h_a = h_a'$, $ h_b = e^{-s} \odot (h_b' - t)$ where $s, t$ are recomputed from $h_a'$.
- Jacobian: block-triangular with identity on the $h_a$ block and diagonal on the $h_b$ block. Therefore

$$ \log\big|\det J_g(h)\big| = \sum_{i=1}^{d_b} s_i. $$

- Permutations: to ensure all dimensions are eventually transformed, alternate partitions (e.g., checkerboard/channel masks) or apply invertible permutations between layers; a popular choice is invertible $1\!\times\!1$ convolutions which implement learned channel permutations with $\log|\det|$ equal to the spatial size times the log-determinant of the channel mixing matrix.

- Pros: fast sampling and density evaluation; parallel across dimensions; easy inverses; stable training.
- Cons: expressivity limited per layer; requires masking/permutes to mix information.

Variant (Affine vs Additive): Additive coupling uses $h_b' = h_b + t$ so the log-determinant is zero; affine coupling uses scale-and-shift as above for nonzero log-determinant and greater flexibility.

### 2) Autoregressive Flows

Autoregressive flows transform each component conditioned on previous ones in a fixed ordering. Let $h = (h_1, \ldots, h_d)$. A general affine autoregressive layer is

$$ h_i' = \mu_i(h_{< i}) + \sigma_i(h_{< i})\,h_i, \quad i = 1, \ldots, d, $$

where $\mu_i, \sigma_i$ are outputs of a masked network ensuring dependence only on $h_{< i}$. The Jacobian is lower triangular with diagonal entries $\sigma_i$.

- Log-determinant:

$$ \log\big|\det J_g(h)\big| = \sum_{i=1}^d \log |\sigma_i(h_{< i})|. $$

- Inverse exists in closed form and can be computed sequentially in the forward or reverse direction depending on parameterization.

Two important directions (MAF vs IAF):

- Masked Autoregressive Flow (MAF): parameterizes $p(x)$ with tractable likelihood evaluation in one pass (compute $\sigma_i, \mu_i$ from $x_{<i}$), but sampling is sequential (one dimension at a time).
- Inverse Autoregressive Flow (IAF): parameterizes the inverse transform for fast parallel sampling from $z \to x$, but likelihood evaluation requires sequential inversion.

- Pros: very expressive per layer; exact log-likelihoods; flexible conditioning structures via masking.
- Cons: either sampling or density evaluation is sequential (trade-off between MAF and IAF); masks constrain parallelism in one direction.

Remark (Triangular Jacobian): For autoregressive transformations the Jacobian structure is strictly triangular, so computing $\log|\det J|$ reduces to summing the logs of diagonal terms, avoiding any expensive determinant.

### 3) Residual Flows (Invertible Residual Networks)

Residual flows use residual layers of the form

$$ h' = h + f(h), $$

with $f$ a Lipschitz-constrained neural network satisfying $\mathrm{Lip}(f) < 1$. Under this contraction condition, the Banach fixed-point theorem implies that the mapping is bijective and that the inverse can be found by fixed-point iteration on $h = h' - f(h)$.

- Inverse: compute $h$ from $h'$ as the unique fixed point of $\phi(h) = h' - f(h)$, e.g., via iterative refinement $h^{(t+1)} = h' - f(h^{(t)})$.

- Log-determinant: using matrix identities,

$$ \log\big|\det J_g(h)\big| = \log\big|\det (I + J_f(h))\big| = \mathrm{Tr}\,\log(I + J_f(h)). $$

With $\|J_f\| < 1$, expand

$$ \mathrm{Tr}\,\log(I + J_f) = \sum_{k=1}^{\infty} \frac{(-1)^{k+1}}{k}\,\mathrm{Tr}\big((J_f)^k\big). $$

Each trace term can be estimated efficiently via Hutchinson’s estimator using a random probe vector $v$ with $\mathbb{E}[vv^\top] = I$:

$$ \mathrm{Tr}(A) = \mathbb{E}_v\big[ v^\top A v \big]. $$

This enables unbiased stochastic estimates of \(\log|\det(I + J_f)|\) without forming Jacobians explicitly; vector–Jacobian products suffice and can be computed via autodiff. In practice, a finite K-term truncation with multiple probes balances bias-variance and cost.

- Pros: highly expressive with free-form Jacobians; avoids coupling/masking constraints; preserves parallelism for both sampling and likelihood (modulo fixed-point and trace estimation costs).
- Cons: requires Lipschitz control for invertibility; log-determinant estimated stochastically; inverse may need iterative solves.

#### Architectural Trade-offs

- Coupling: simple inverses and cheap exact $\log|\det J|$; requires permutations/masks for mixing.
- Autoregressive: triangular Jacobians with exact $\log|\det J|$; sequential in one direction (sampling or likelihood) depending on parameterization.
- Residual: most flexible Jacobians; relies on contraction for invertibility and stochastic trace/logdet estimation.

#### Practical Notes Referenced in the Presentation

- Alternating masks or learned invertible $1\!\times\!1$ convolutions to permute/mix channels between coupling layers; the latter has $\log|\det|$ equal to spatial size times the log-determinant of the channel-mixing matrix.
- For autoregressive layers, masked networks (e.g., MADE-style masking) enforce causal structure so that $h_i'$ depends only on $h_{<i}$.
- For residual flows, spectral normalization or other Lipschitz constraints ensure $\|J_f\| < 1$ to guarantee bijectivity and enable series expansions.

#### References

- Video notes: [How I Understand Flow Matching — Jia-Bin Huang](https://youtubetodoc.s3.amazonaws.com/docs/youtube/DDq_pIfHqLs.md)
- Survey: [Papamakarios et al., “Normalizing Flows for Probabilistic Modeling and Inference”](https://arxiv.org/abs/1912.02762)
- Survey: [Kobyzev et al., “Normalizing Flows: An Introduction and Review of Current Methods”](https://arxiv.org/abs/1908.09257)


