# Problem formulation — and a unifying lens for how models actually sample

We observe $x\in\mathcal{X}$ drawn from an unknown law $p_{\text{data}}$. A generative model specifies a family $p_\theta$ and a concrete sampler $\mathsf{Sample}_\theta$ that can emit $x\sim p_\theta$. Think of the object as $(p_\theta,\mathsf{Sample}_\theta)$.

Training fits $\theta$ by minimizing a divergence between $p_{\text{data}}$ and $p_\theta$. Maximum likelihood is

$$
\theta^\star \in \arg\min_\theta
\mathrm{KL}\big(p_{\text{data}} \,|\, p_\theta\big)
= \arg\min_\theta
\mathbb{E}_{x\sim p_{\text{data}}}\big[-\log p_\theta(x)\big].
$$

Empirically, replace the expectation by the average over $\{x_i\}_{i=1}^N$.

Sampling is separate from likelihood. It is a **map** or a **Markov kernel**. Formally, $\mathsf{Sample}_\theta$ is either a measurable function $T_\theta$ with $x=T_\theta(z)$, $z\sim \mu$, or a transition law $K_\theta(\cdot\mid\cdot)$ that evolves a chain to stationarity.

This separation matters because families mostly differ in the **sampling operator**. Four mechanisms cover modern practice: **factorize**, **transform**, **evolve**, **shape**. Each gives a distinct training loss and runtime profile.

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
\log p_\theta(x)=\log p_Z\big(T_\theta^{-1}(x)\big)+\log\big|\det J_{T_\theta^{-1}}(x)\big|.
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

**Why this lens?** Because inference constraints pick the mechanism. Exact likelihoods favor factorize/transform; ultra-low latency favors one-shot transforms or distilled evolutions; top fidelity and flexible conditioning favor evolutions; unnormalized scientific targets favor shaping plus sampler dynamics.

Formally, each mechanism chooses a **pushforward**: a triangular map (factorize), a diffeomorphism (transform), a time-indexed flow of measures $p_t$ (evolve), or an invariant kernel $K_\theta$ (shape). The rest—losses, compute, and quality–latency trade-offs—fall out of that choice.
