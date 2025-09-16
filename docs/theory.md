# 📐 **Autoregressive Flows for Cross-Temperature Transport**

## **Problem Setup**

**Configuration Space**: $Ω = ℝ^{3N}$ (Cartesian coordinates for N atoms)

**Base Distribution**: $P_{β_{i}}$ (Boltzmann distribution at temperature $β_i$)

**Target Distribution**: $P_{β_{i+1}}$ (Boltzmann distribution at adjacent temperature $β_{i+1}$)

$$p_β(x) = Z_β^{-1} \exp(-βU(x))$$

**Goal**: Learn a bijection $T_{\theta}: Ω → Ω$ for all temperature pairs in replica ladder via a *transferable* autoregressive normalizing flow for cross-temperature transport $P_{β_k} → P_{β_{k+1}}$, enabling flow-enhanced PT with higher acceptance rates across unseen peptides.

## **Autoregressive Flow Architecture**

**Flow $T_{\theta}$**: Composition of autoregressive transformations $T_{\theta} = T_{3N} ∘ ... ∘ T_{2} ∘ T_{1}$

**Individual Transformations $T_i$**: Each $T_i$ transforms coordinate $i$ conditioned on previous coordinates:
$$z_i = T_i(x_i; x_{<i}), \quad i = 1, ..., 3N$$

**Triangular Jacobian**: Autoregressive structure yields lower triangular Jacobian matrix
$$\det(∂T_{\theta}/∂x) = \prod_{i=1}^{3N} ∂T_i/∂x_i$$

This factorization is useful because it enables exact likelihood computation via change of variables

**Change of Variables Formula**:
$$p_{\beta_{i+1}}(x_{\beta_{i+1}}) = p_{\beta_i}(x_{\beta_i}) \left|\det\left(\frac{∂T_{\theta}}{∂x_{\beta_i}}\right)\right|^{-1}$$

## **Flow-Enhanced PT**

**Standard PT Swap**: $(x_{\beta_i}, x_{\beta_{i+1}}) → (x_{\beta_{i+1}}, x_{\beta_i})$

**Transport Swap**: Replace naive swap with flow-aligned proposal:
$$g(x_{\beta_i}, x_{\beta_{i+1}}) = (T_{\theta}^{-1}(x_{\beta_{i+1}}), T_{\theta}(x_{\beta_i}))$$

**Enhanced Acceptance**:
$$α_{flow}(x_{\beta_i}, x_{\beta_{i+1}}) = \min\{1, \exp(Δ_{flow}(x_{\beta_i}, x_{\beta_{i+1}}))\}$$

**Acceptance Ratio** (derived from detailed balance):
$$Δ_{flow}(x_{\beta_i}, x_{\beta_{i+1}}) = -β_i U(T_{\theta}^{-1}(x_{\beta_{i+1}})) - β_{i+1} U(T_{\theta}(x_{\beta_i})) + β_i U(x_{\beta_i}) + β_{i+1} U(x_{\beta_{i+1}}) + \log|\det J_{T_{\theta}}(x_{\beta_i})| + \log|\det J_{T_{\theta}^{-1}}(x_{\beta_{i+1}})|$$

## **Training Objective**

**Bidirectional Transport**:
$$x_{\beta_i} \sim P_{β_i} \Rightarrow T_{\theta}(x_{\beta_i}) \sim P_{β_{i+1}}, \quad x_{\beta_{i+1}} \sim P_{β_{i+1}} \Rightarrow T_{\theta}^{-1}(x_{\beta_{i+1}}) \sim P_{β_i}$$

**NLL Loss** (maximizes swap acceptance):
$$L_{NLL}(θ) = \mathbb{E}_{x_{\beta_{i+1}} \sim p_{β_{i+1}}}[β_i U(T_{\theta}^{-1}(x_{\beta_{i+1}})) - \log|\det J_{T_{\theta}^{-1}}(x_{\beta_{i+1}})|] + \mathbb{E}_{x_{\beta_i} \sim p_{β_i}}[β_{i+1} U(T_{\theta}(x_{\beta_i})) - \log|\det J_{T_{\theta}}(x_{\beta_i})|]$$

**Connection to PT**: Minimizing $L_{NLL}(θ)$ directly maximizes $Δ_{flow}(x_{\beta_i}, x_{\beta_{i+1}})$, leading to higher swap acceptance rates and enhanced sampling efficiency.

## **Temperature Conditioning**

**Multiple Models**: Train separate $T_{\theta}^{(i,i+1)}$ for each temperature pair

**Conditional Architecture**: Embed temperatures directly into transport:
$$T_{\theta}(x_{\beta_i}, β_i, β_{i+1}) → x_{\beta_{i+1}}$$

**Collapse**: $N-1$ pairwise models → single conditional map

## **Transferability**

**Challenge**: Train on {AA,AK,AS} → generalize to {SA,SK,SS,KK,KS}

**Architecture Requirements**: Universal molecular physics representation in 3N-dimensional space