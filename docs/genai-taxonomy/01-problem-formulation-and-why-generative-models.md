# Problem Formulation: The Generative Modeling Challenge

## The Fundamental Question

Given finite observations $\{x_i\}_{i=1}^N$ sampled from some unknown distribution $p_{\text{data}}(x)$ over space $\mathcal{X}$, can we build a computational procedure that generates new, realistic samples from this same distribution?

This is the **generative modeling problem**: learning to synthesize data that is indistinguishable from the training distribution, despite never having direct access to $p_{\text{data}}$ itself.

## Why This Matters

The ability to generate realistic data is fundamental to understanding complex systems:
- **Scientific modeling**: Simulating molecular conformations, weather patterns, or economic scenarios
- **Creative synthesis**: Generating text, images, music, or code
- **Data augmentation**: Creating training examples for downstream tasks
- **Uncertainty quantification**: Understanding the space of plausible outcomes

## The Mathematical Framework

A generative model consists of two coupled components:

1. **Density model**: A parametric family $p_\theta(x)$ that approximates $p_{\text{data}}(x)$
2. **Sampling mechanism**: A procedure $\mathsf{Sample}_\theta$ that can draw $x \sim p_\theta$

The generative model is the pair $(p_\theta, \mathsf{Sample}_\theta)$.

### Training: Learning the Distribution

We fit parameters $\theta$ by minimizing a divergence between $p_{\text{data}}$ and $p_\theta$. Maximum likelihood estimation targets:

$$
\theta^\star \in \arg\min_\theta \mathrm{KL}\big(p_{\text{data}} \,\|\, p_\theta\big) = \arg\min_\theta \mathbb{E}_{x \sim p_{\text{data}}}\big[-\log p_\theta(x)\big]
$$

Empirically, this becomes minimizing negative log-likelihood over the dataset: $-\frac{1}{N}\sum_{i=1}^N \log p_\theta(x_i)$.

### The Sampling-Likelihood Duality

Here lies the central tension: **sampling and likelihood evaluation are dual computational problems**. 

If we can efficiently sample from $p_\theta$, computing $p_\theta(x)$ is generally hard. If we can tractably evaluate $p_\theta(x)$, sampling is often expensive. This duality forces fundamental trade-offs.

Formally, $\mathsf{Sample}_\theta$ is either:
- A **deterministic map**: $x = T_\theta(z)$ where $z \sim \mu$ (simple base distribution)
- A **stochastic process**: Markov kernel $K_\theta(x' \mid x)$ that converges to $p_\theta$

## The Four Fundamental Mechanisms

All modern generative models resolve the sampling-likelihood tension through one of four core mechanisms. Each embodies a different computational philosophy:

**Factorize** → Sequential decomposition via chain rule  
**Transform** → Invertible coordinate changes  
**Evolve** → Stochastic dynamics over time  
**Shape** → Energy landscapes with equilibrium sampling  

Each mechanism induces distinct training objectives, computational profiles, and quality-efficiency trade-offs. The choice determines everything: from the loss function to the inference algorithm.

---

**Factorize.** We commit to the chain rule and model

$$
p_\theta(x)=\prod_{t=1}^T p_\theta\big(x_t \mid x_{<t}\big).
$$

Log-likelihood is cross-entropy; sampling draws $x_1,\dots,x_T$ sequentially.

The sampler is a triangular “Knothe–Rosenblatt–style” transport: if $u_t\sim \mathrm{Unif}(0,1)$,

$$
x_t = F_\theta^{-1}\big(u_t \mid x_{<t}\big),
$$

a stepwise pushforward from $[0,1]^T$ to sequences. For discrete tokens this is piecewise-constant.

This gives exact NLL and calibrated perplexity, at the cost of $O(T)$ sampling latency and exposure bias during generation.

---

**Transform.** We learn a deterministic pushforward $x=T_\theta(z)$ from a simple base $z\sim \mu$. If $T_\theta$ is a diffeomorphism (flows/CNFs),

$$
\log p_\theta(x)=\log \mu\big(T_\theta^{-1}(x)\big)+\log\big|\det J_{T_\theta^{-1}}(x)\big|.
$$

Training is exact MLE; sampling is one shot: draw $z$, compute $T_\theta(z)$.

In continuous time (CNFs), $x_t$ solves $\dot x_t=v_\theta(x_t,t)$ and densities obey the instantaneous change of variables:

$$
\frac{d}{dt}\log p_t(x_t)= -\,\nabla\cdot v_\theta(x_t,t).
$$

Log-likelihood reduces to an ODE integral of $-\nabla\cdot v_\theta$.

If we drop tractable likelihoods and train a generator adversarially, the map is $x=G_\theta(z)$ with a min–max loss, e.g. Wasserstein:

$$
\min_\theta \ \max_{\|f\|_L\le 1}\ \mathbb{E}_{x\sim p_{\text{data}}}[f(x)]-\mathbb{E}_{z\sim\mu}[f(G_\theta(z))].
$$

Sampling stays one shot; likelihood becomes implicit.

---

**Evolve.** We move probability mass along time from a simple prior to data. A generic SDE

$$
dx_t = v_\theta(x_t,t)\,dt + \sigma(t)\,dW_t,\qquad t\in[0,1]
$$

induces a path of marginals $p_t$. The reverse-time SDE (or the probability-flow ODE) is learned.

Score-based diffusion fits the score $s_\theta(x,t)\approx \nabla_x \log p_t(x)$ via denoising:

$$
\min_\theta\ \mathbb{E}_{t,x,\varepsilon}\!\left[ \lambda(t)\,\big\|s_\theta(x_t,t)-\nabla_x\log p_t(x_t)\big\|^2 \right]
\approx
\mathbb{E}\!\left[ \lambda(t)\,\big\|s_\theta(x_t,t)+\sigma(t)^{-1}\varepsilon\big\|^2 \right].
$$

Sampling integrates the learned reverse dynamics; ODE variants yield likelihoods via

$$
\log p_1(x_1)=\log p_0(x_0)-\int_0^1 \nabla\cdot v_\theta(x_t,t)\,dt.
$$

Distillation can compress many steps into few or one, approximating a direct transport.

---

**Shape.** We specify an energy $E_\theta$ and define $p_\theta(x)\propto e^{-E_\theta(x)}$. The sampler is a Markov chain with $p_\theta$ stationary, e.g. Langevin:

$$
x^{k+1}=x^k-\eta\,\nabla_x E_\theta(x^k)+\sqrt{2\eta}\,\xi^k,\qquad \xi^k\sim\mathcal{N}(0,I).
$$

Gradients of log-density are scores: $\nabla_x \log p_\theta(x)=-\nabla_x E_\theta(x)$. Hyvärinen score matching fits scores without normalizers:

$$
\min_\theta\ \mathbb{E}_{x\sim p_{\text{data}}}\big\|\nabla_x \log p_\theta(x)-\nabla_x \log p_{\text{data}}(x)\big\|^2,
$$

often approximated by denoising objectives. Mixing time and partition functions are the practical hurdles.

---

## The Unifying Perspective

Each mechanism is fundamentally a different **pushforward strategy**—a way to transport probability mass from a simple space where sampling is trivial to the complex data space:

- **Factorize**: Triangular transport via sequential conditioning
- **Transform**: Direct diffeomorphic pushforward  
- **Evolve**: Time-indexed stochastic transport
- **Shape**: Stationary distribution of ergodic dynamics

The computational trade-offs emerge naturally:
- **Exact likelihoods** → factorize/transform
- **Ultra-low latency** → one-shot transforms or distilled flows  
- **Highest fidelity** → evolution-based methods
- **Unnormalized targets** → energy-based shaping

This taxonomy is exhaustive: every modern generative model implements one of these four transport mechanisms, and each mechanism uniquely determines the training objective, sampling procedure, and computational characteristics.
