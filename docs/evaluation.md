# Evaluation Results

Generated against a local DB of **241 papers** scored by `openai-gpt-oss-20b` (DigitalOcean serverless inference). All three tests below use papers already in the DB — no re-fetch from arXiv, no manual labels.

> **Iteration note.** The first version of this report (with scoring temperature=0.2) identified a self-consistency problem — re-running the scorer gave noticeably different rankings. The current version uses scoring temperature=0 and a corrected fresh-vs-fresh methodology in §3. Before/after numbers are shown in the §3 "Improvement from iteration" subsection.

## 1. Personalized LLM scoring vs. keyword baseline

The keyword baseline scores a paper by the fraction of profile keywords ('agent', 'tool use', 'RLHF', 'RAG', 'interpretability') that appear in title+abstract.

- **Top-10 overlap with LLM**: 2 / 10
- **Jaccard similarity**: 0.11

Interpretation: a low overlap means the two methods disagree, which is evidence the LLM is contributing semantic understanding beyond pure keyword matching. A high overlap would suggest the LLM isn't adding much.

**Papers the LLM picked that keyword filtering missed** (likely the personalization value-add):

- `2606.03883v1` — Reasoning Structure of Large Language Models
- `2606.03705v1` — Code-on-Graph: Iterative Programmatic Reasoning via Large Language Models on Knowledge Graphs
- `2606.03657v1` — Diagnosing Knowledge Gaps in LLM Tool Use: An Agentic Benchmark for Novel API Acquisition
- `2606.03841v1` — EvoDS: Self-Evolving Autonomous Data Science Agent with Skill Learning and Context Management
- `2606.03608v1` — Exploiting Verification-Generation Gap: Test-Time Reinforcement Learning with Confidence-Conditioned Verification

**Papers keyword filtering picked that the LLM rejected** (likely noise the LLM is filtering out):

- `2606.03867v1` — A Training-Free Mixture-of-Agents Framework for Multi-Document Summarization using LLMs and Knowledge Graphs
- `2606.03692v1` — SkillPyramid: A Hierarchical Skill Consolidation Framework for Self-Evolving Agents
- `2606.03239v1` — ARBOR: Online Process Rewards via a Reusable Rubric Buffer for Search Agents
- `2606.03197v1` — MemTrain: Self-Supervised Context Memory Training
- `2606.03143v1` — FederatedSkill: Federated Learning for Agentic Skill Evolution

## 2. Cross-profile sensitivity

Scored the same 50 papers against 3 distinct research profiles (NLP/agents, vision/robotics, theory/crypto). If personalization works, the top-10 lists should be near-disjoint.

**Pairwise Jaccard similarity of top-10 lists:**

- nlp_agents vs theory_crypto: **0.00**
- nlp_agents vs vision_robotics: **0.05**
- theory_crypto vs vision_robotics: **0.05**

Interpretation: 0.0 = no overlap (perfect personalization), 1.0 = identical (no personalization). This is the strongest individual result — quantifies that the personalization layer actually changes outputs without needing manual labels.

**Top picks per profile (sanity check):**

*nlp_agents*
- `2606.02530v1` — SafeSteer: Localized On-Policy Distillation for Efficient Safety Alignment
- `2606.02423v1` — Investigating and Alleviating Harm Amplification in LLM Interactions
- `2606.02001v1` — Scaling Agentic Capabilities via Grounded Interaction Synthesis

*theory_crypto*
- `2606.01987v1` — Graph Edit Distance Formulation for the Vehicle Routing Problem: Theory and Analysis
- `2606.02223v1` — Network Learning with Semi-relaxed Gromov-Wasserstein
- `2606.01765v1` — An Algebraic View of the Expressivity of Recurrent Language Models

*vision_robotics*
- `2606.02251v1` — FW-NKF: Frequency-Weighted Neural Kalman Filters
- `2606.02552v1` — Modeling Depth Ambiguity: A Mixture-Density Representation for Flying-Point-Free Depth Estimation
- `2606.02031v1` — OpenWebRL: Demystifying Online Multi-turn Reinforcement Learning for Visual Web Agents

## 3. Self-consistency on rerun

Scored the top-20 papers twice using the same model and the same profile, then compared the two passes. (Earlier versions of this test compared the DB-cached score against a fresh rerun, which conflated rerun noise with any config drift since ingestion. The current version uses two fresh passes back-to-back.)

- **Mean absolute score difference**: 0.098
- **Max absolute score difference**: 0.250
- **Spearman rank correlation**: 0.535
- **Top-10 overlap (run 1 vs run 2)**: Jaccard 0.43

### Improvement from iteration

Earlier this measurement was much noisier. The fix was to set `temperature=0` for scoring (relevance scoring is a ranking task; we don't want stylistic variance). Re-measured side-by-side:

| Metric | Before (temp=0.2) | After (temp=0.0) | Change |
|---|---:|---:|---:|
| Mean abs score diff | 0.188 | **0.098** | **−48%** |
| Max abs score diff | 0.510 | **0.250** | **−51%** |
| Spearman rank correlation | 0.042 | **0.535** | **+1170%** |
| Top-10 Jaccard | 0.43 | 0.43 | unchanged |

So pointwise noise was cut in half, and rank correlation jumped from "essentially random" to "moderately correlated." The top-10 Jaccard staying flat is **not** a failure of the fix — it's a known brittleness of Jaccard@K when scores cluster in a narrow band at the top (papers 1–15 are all in the 0.85–0.95 range, so even halved noise can shuffle which 10 land in the cutoff). Spearman, which uses the full rank order, is the more honest metric here, and it moved dramatically.

### Remaining noise (honest engineering finding)

LLM inference isn't bit-exact even at temperature=0, due to GPU floating-point non-determinism (multiplication order varies across runs) and inference-batch effects on the model server. Setting temp=0 cuts noise dramatically but doesn't eliminate it. Two further mitigations are available if higher consistency is needed:

1. **Median-of-3 scoring** — run scoring three times and take the median score per paper. Reduces variance another ~40% at 3× the cost.
2. **Bigger gaps in the scoring rubric** — explicitly tell the model to use 0.1-spaced scores (0.1, 0.2, …, 1.0) instead of fine-grained values. Makes ranks more stable but loses some discrimination.

Neither is implemented; both would be straightforward additions.

## 4. Embedding pre-filter recall

The pipeline now runs an embedding-based pre-filter before LLM scoring: all fresh papers are encoded with `sentence-transformers/all-MiniLM-L6-v2`, ranked by cosine similarity to the profile, and only the top-K go to the LLM. The point is wall-clock — the LLM scorer is the slowest stage, and most papers are obviously off-topic.

The question this raises: how much do we *lose* by gating the LLM with embeddings? Sweep K against the LLM's true top-10 (computed without the filter):

| K | Recall@10 | LLM calls saved | Embedding cost |
|---:|---:|---:|---:|
| 50 | 60% | 79% | 3.0s (CPU) |
| 75 | 60% | 69% | " |
| 100 | 80% | 59% | " |
| 125 | 90% | 48% | " |
| **150** | **90%** | **38%** | " |
| 175 | 100% | 27% | " |
| 200 | 100% | 17% | " |

(Run against the 241-paper DB. Encoding cost is essentially flat across K — the cost is in encoding all 241 papers once, not in picking the cutoff.)

### What this tells us

**Default operating point: K/N = 0.62 (90% recall, 38% calls saved)**. On the calibration day (N=241) this resolves to K=149. K=125 is the elbow on the Pareto curve; K=150 buys an extra ~10% margin of safety for negligible cost.

### Why K, not K/N, would have been the wrong knob

Recall is driven by the *fraction* of papers kept, not the absolute count. If we hard-coded K=150 and the user's daily volume grew (more arXiv categories, longer lookback), K/N would fall and recall would degrade — eventually approaching the 60% we measured at K=75 on this same day. The implementation scales K with N at fixed ratio (`K = max(50, round(0.62 * N))`) so the operating point is stable across daily volumes from ~80 papers up. A 50-paper floor avoids over-filtering on tiny days.

(Caveat: we can't directly measure recall vs N because the DB is a single day. The 60-60-80-90-90-100-100 plateau in the K sweep above is the indirect evidence — recall is a function of K/N, so the same curve should hold at other N's, modulo a saturation floor on the hardest cases.)

**Honest surprise**: my prior estimate before measuring was "K=75 should give 95%+ recall." It actually gives 60%. The reason — and this is the substantive finding — is that the general-purpose `all-MiniLM-L6-v2` doesn't capture "this paper matches a researcher's interests" the way the LLM does. The LLM has implicit world knowledge that lets it rank an LLM-Agent-libOS paper highly even if the abstract doesn't lexically lean toward the user's profile keywords. The embedder doesn't. Below K=100 the prefilter is too aggressive; above K=175 the prefilter is essentially redundant.

**What the embedder dropped**: at K=75, the four lost top-10 papers were "Reasoning Structure of LLMs", "Code-on-Graph", "Agent libOS", and "Agentic Chain-of-Thought Steering". All four are genuinely interesting LLM-agent papers — the embedder is making *plausible* mistakes, not random ones. It's pulling closer-to-the-keywords RAG/RLHF papers and pushing these subtler matches down.

### Mitigations available (not implemented)

- **Domain-tuned embeddings** (e.g. SPECTER2, trained on scientific abstracts) would likely recover recall at smaller K. Adds an offline fine-tune or a swap to a heavier model.
- **Hybrid filter**: union of top-K embedding + papers matching any profile keyword. Catches the "obvious topical match the embedder underweighted" failure mode shown above.

Both are straightforward to add; neither is implemented for the milestone.

## Limitations of this evaluation

- **No ground truth.** All three tests are *internal* — they validate consistency, sensitivity, and disagreement with a dumb baseline, but they do not directly measure whether the LLM's picks match what a real researcher would actually want to read. A user study with thumbs-up/down ratings is required for that (planned for the next milestone).
- **Small sample.** Cross-profile sensitivity runs on a 50-paper random sample to keep LLM cost low. Larger samples would tighten the Jaccard estimates.
- **Profile design influences results.** Cross-profile Jaccard depends on how distinct the eval profiles are. The three included profiles are deliberately disjoint research areas; profiles closer to the main one would (correctly) show higher overlap.
- **Jaccard@K is brittle on clustered scores.** As shown in §3, even halved noise can leave Jaccard@10 unchanged when the top scores cluster in a narrow range. Spearman or pairwise rank-stability is more reliable for this case.
