## Evaluation & Diagnostics

Evaluation depends on the paradigm and application. Use complementary views: likelihoods, sample quality/diversity, task and human measures, and calibration/coverage.

### Likelihood-based
- NLL / bits-per-dimension; perplexity for discrete AR models
- ODE likelihoods or change-of-variables for flows/CNFs

### Sample Quality and Diversity
- FID \([Heusel et al., 2017](https://arxiv.org/abs/1706.08500)) and KID \([Binkowski et al., 2018](https://arxiv.org/abs/1801.01401))
- Precision–recall for generative models \([Sajjadi et al., 2018](https://arxiv.org/abs/1806.00035); [Kynkäänniemi et al., 2019](https://arxiv.org/abs/1904.06991))

### Task-based & Human Evaluation
- Instruction following and preference modeling for text and multimodal systems
- Pairwise human judgments and rubric-based scoring

### Calibration & Coverage
- Reliability diagrams; coverage under constraints and selective prediction
- Out-of-distribution detection; ablations on guidance strength and conditioning

### Text-specific Considerations
- Toxicity bias: RealToxicityPrompts \([Gehman et al., 2020](https://arxiv.org/abs/2009.11462))
- Factuality: TruthfulQA \([Lin et al., 2021](https://arxiv.org/abs/2109.07958))
- Long-range coherence: narrative consistency and retrieval augmentation stress tests

Continue: [Controllability & Conditioning](./09-controllability-and-conditioning.md)


