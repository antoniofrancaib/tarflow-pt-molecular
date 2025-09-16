### Change of Variables via Local Volume Preservation

This note derives the change-of-variables formula in a way that mirrors the elegant, geometric argument from the referenced presentation, emphasizing probability mass preservation and local linearization. For context and motivation, see the lecture notes and video summary by Jia-Bin Huang: [How I Understand Flow Matching](https://youtubetodoc.s3.amazonaws.com/docs/youtube/DDq_pIfHqLs.md). For background surveys on normalizing flows, see [Papamakarios et al. (2021)](https://arxiv.org/abs/1912.02762) and [Kobyzev et al. (2020)](https://arxiv.org/abs/1908.09257).

#### Setup

- We have a base random variable $Z \in \mathbb{R}^d$ with density $p_Z$, and an invertible, differentiable map $g: \mathbb{R}^d \to \mathbb{R}^d$.
- We define $X = g(Z)$. Our goal is to express the density $p_X$ in terms of $p_Z$ and the Jacobian of $g$ (or of $g^{-1}$).
- The key principle is probability mass preservation under bijective change of variables: for small regions $A$ and their images $g(A)$, we have $\mathbb{P}[Z \in A] = \mathbb{P}[X \in g(A)]$.

#### One-dimensional derivation (d = 1)

Let $y = g(z)$ be strictly monotone and differentiable. For a small interval $[z, z+dz]$, its image is approximately $[y, y+dy]$ with $dy = g'(z)\,dz$. By mass preservation:

$$ p_Y(y)\,dy = p_Z(z)\,dz. $$

Hence

$$ p_Y(y) = p_Z(z)\,\left|\frac{dz}{dy}\right| = p_Z\big(g^{-1}(y)\big)\,\left|(g^{-1})'(y)\right|. $$

This is the 1D change-of-variables formula. The absolute value appears because orientation-reversing mappings $g'(z) < 0$ still preserve nonnegative densities.

#### Multivariate derivation (d ≥ 2)

Let $x = g(z)$ with $g$ a $C^1$ diffeomorphism. Fix a point $z_0$ and denote $x_0 = g(z_0)$. Consider a small hyper-rectangle around $z_0$ with edge vectors forming the columns of a matrix $\Delta Z$. By first-order Taylor expansion, locally

$$ g(z_0 + \delta z) \approx g(z_0) + J_g(z_0)\,\delta z, $$

where $J_g(z_0) = \frac{\partial g}{\partial z}(z_0)$ is the Jacobian matrix. Therefore, the image of the small hyper-rectangle under $g$ is approximately a parallelotope with edge matrix $\Delta X = J_g(z_0)\,\Delta Z$. Volumes scale by the absolute determinant:

$$ |\Delta X| = |\det J_g(z_0)|\,|\Delta Z|. $$

Mass preservation over these corresponding regions gives

$$ p_X(x_0)\,|\Delta X| = p_Z(z_0)\,|\Delta Z|. $$

Canceling $|\Delta Z|$ and substituting the volume relation yields

$$ p_X(x_0) = \frac{p_Z(z_0)}{|\det J_g(z_0)|}. $$

Since $z_0 = g^{-1}(x_0)$, and using the identity

$$ J_{g^{-1}}(x_0) = J_g(z_0)^{-1} \quad \Rightarrow \quad |\det J_{g^{-1}}(x_0)| = \frac{1}{|\det J_g(z_0)|}, $$

we obtain the standard multivariate change-of-variables formula:

$$ p_X(x) = p_Z\big(g^{-1}(x)\big)\,\big|\det J_{g^{-1}}(x)\big|. $$

Equivalently,

$$ p_X\big(g(z)\big) = \frac{p_Z(z)}{|\det J_g(z)|}. $$

#### Composition of layers and log-likelihood

Normalizing flows use compositions of simple, invertible maps to build flexible transformations. Let

$$ x = g_K \circ g_{K-1} \circ \cdots \circ g_1(z), \quad z \sim p_Z. $$

Define the intermediate variables $h_0 = z$ and $h_k = g_k(h_{k-1})$. Applying the formula layer by layer gives

$$ p_X(x) = p_Z(z) \prod_{k=1}^K \frac{1}{\big|\det J_{g_k}(h_{k-1})\big|}. $$

Taking logs:

$$ \log p_X(x) = \log p_Z(z) - \sum_{k=1}^K \log \big|\det J_{g_k}(h_{k-1})\big|. $$

Equivalently, in terms of the inverse Jacobians along the forward pass $h_k$:

$$ \log p_X(x) = \log p_Z(z) + \sum_{k=1}^K \log \big|\det J_{g_k^{-1}}(h_k)\big|. $$

These identities are the backbone of maximum likelihood training in normalizing flows: the likelihood decomposes into a base-density term plus tractable Jacobian-determinant corrections tailored by the architecture.

#### A quick sanity check (1D affine map)

Let $y = a z + b$ with $a \neq 0$. Then $z = (y - b)/a$ and $dz/dy = 1/a$. The formula gives

$$ p_Y(y) = p_Z\!\left(\frac{y-b}{a}\right)\,\frac{1}{|a|}, $$

exactly matching the intuition that stretching by $|a|$ compresses density by $1/|a|$.

#### Intuition in one sentence

Locally, every smooth invertible mapping is approximately linear; linear maps scale volumes by $|\det J|$; probability mass stays the same, so density must scale inversely by $|\det J|$.

#### References

- Video notes: [How I Understand Flow Matching — Jia-Bin Huang](https://youtubetodoc.s3.amazonaws.com/docs/youtube/DDq_pIfHqLs.md)
- Survey: [Papamakarios et al., “Normalizing Flows for Probabilistic Modeling and Inference”](https://arxiv.org/abs/1912.02762)
- Survey: [Kobyzev et al., “Normalizing Flows: An Introduction and Review of Current Methods”](https://arxiv.org/abs/1908.09257)


