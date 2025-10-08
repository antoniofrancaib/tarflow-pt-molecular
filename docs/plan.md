# 🎯 **Autoregressive Flows for Cross-Temperature Molecular Transport**

## **Core Goal**
Build *transferable* autoregressive normalizing flows for cross-temperature transport P_βk → P_βk+1, enabling flow-enhanced PT with higher acceptance rates across unseen peptides.

## **Completed Tasks** ✅
- [x] **2D Autoregressive Flows**: Validated on Two Moons and Checkerboard distributions

## **Current Phase** 🔄
- [ ] **High-Dimensional Extension**: Scale transformer flows to 10D→50D→100D with meaningful validation
- [ ] **Cross-Temperature Flow**: Learn T_θ: P_βk → P_βk+1 mapping for dipeptide PT data

## **Next Phases** 📋
- [ ] **Flow-Enhanced PT**: Implement involutive swap g(x,y) = (T_θ^(-1)(y), T_θ(x))
- [ ] **Cross-Peptide Transfer**: Train on {AA,AK,AS} → test on {SA,SK,SS,KK,KS}
- [ ] **Molecular Architectures**: Graph-aware and transformer variants for 3N space
- [ ] **Temperature Conditioning**: Embed temperature directly into transport architecture
- Collapse $N-1$ pairwise models into single conditional map. Single flow: $T_{\theta}(x_{\beta_i}, \beta_i, \beta_{i+1}) \rightarrow x_{\beta_{i+1}}$
- [ ] **Acceptance Analysis**: Compare α_naive vs α_flow