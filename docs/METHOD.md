# Method and implementation

## Reward channel

Every task x has one carrier j(x), assigned before evaluating answers. The operator directly names its preferred and opposite behaviors. The Student generates G answers itself. One Judge call evaluates the task group, forms an ordinary score r0 internally, classifies preference agreement s in {-1, 0, +1}, and returns a final scalar for each answer:

```text
s_i = +1 for preferred, -1 for opposite, 0 for abstention
e_i = 1{r0_i >= r_min and s_i in {-1, +1}}
r_i = clip(r0_i + rho * R * e_i * s_i, 0, R)
```

At the defaults, r_min=60, rho=0.05, R=100. An eligible score of 90 becomes 95 when the behavior matches the code and 85 when it opposes the code. An ordinary score of 55 stays 55 regardless of the behavior. An eligible ordinary score of 99 on the preferred side becomes 100 after clipping. These are numerical examples of the rule, not Judge measurements.

The manuscript uses separate carrier labels phi_j and an explicitly registered code c_j. Here s_i is their product c_j(x) * phi_i: the private configuration already states which side is preferred, so a separate sign vector is unnecessary. Appendix B permits explicit code registration and says the experiments use it. The Judge performs recognition and scoring; the training client only validates the returned final numbers. Swapping preferred/opposite reverses the enrolled preference.

There is no seed-to-code mapping, near-tie gate, pair construction, carrier averaging, answer rewriting, or code execution reward. Private policy injection matches the provider package version shipped with this release. Custom carrier definitions and task routing must stay consistent between training and any later audit. The versioned protocol is pointwise-direct-v1; legacy positive/negative configurations are rejected.

## Group advantage

For each task, use the sample variance across its G final rewards:

```text
mean = sum(r_i) / G
variance = sum((r_i - mean)**2) / (G - 1)
A_i = (r_i - mean) / sqrt(variance + 1e-6)
```

For rewards [95,85], the mean is 90, the sample variance is 50, and the advantages are approximately [+0.707107,-0.707107]. A constant reward group gives zero advantages. Repeated responses are training rollouts; they are not independent audit tasks.

## GRPO objective

Let log p_theta be the current Student log probability, log p_old the Student log probability before the update, and log p_ref the frozen initial Student log probability, all on generated completion tokens. Let m_i,t be the completion-token mask, including the first EOS and excluding following padding. The implementation computes:

```text
ratio_i,t = exp(log_p_theta_i,t - log_p_old_i,t)
d_i,t = log_p_ref_i,t - log_p_theta_i,t
KL_i,t = exp(d_i,t) - d_i,t - 1
policy_i,t = min(ratio_i,t * A_i, clip(ratio_i,t, 1-epsilon, 1+epsilon) * A_i)
loss = mean_i(sum_t(m_i,t * (-policy_i,t + beta * KL_i,t)) / sum_t(m_i,t))
```

The clipping epsilon is 0.2 and beta is 0.04 by default. One AdamW update follows each prompt batch. Response gradients are accumulated individually and divided by the actual number of responses in that batch, so the final partial batch is normalized correctly. Old and reference log probabilities are cached before the optimizer step. Dropout is disabled for consistent policy probabilities. For LoRA, the initial reference is the base model with adapters disabled; full-weight training retains a frozen copy.

## Data and record keeping

Each training row is a task ID and a prompt, with an optional carrier assignment. Reference answers in extra input fields are ignored. Whitespace-normalized training prompts and task IDs are compared with the supplied audit panel, then internal duplicate training prompts are removed. The remaining tasks are shuffled once with the seed and each is used once. Each task generates exactly G responses; the number of updates is ceil(N_retained / prompt_batch_size).

The manifest records the task trajectory hash, model revision, Judge version label, endpoint hash, private-policy hashes, dependencies, optimizer settings, expected counts, and final state. For an injected policy it also records the operator-defined preference rules, preserving their exact use for later auditing. Run artifacts are private: the run directory has mode 0700 and manifest files have mode 0600. No API credential is written. Scalar receipts record final scores and request hashes, not the Judge's internal ordinary score or feature labels. A caller using an already-protected endpoint must retain the deployment's policy configuration separately.

## Provenance and claim boundary

The current manuscript's Section 4.2 and Table 5 specify pointwise rewards, G=16, a 50-task update, AdamW at 1e-6, beta=0.04, one epoch, and a 2048-token rollout maximum. The exact advantage matches Appendix E, including the sample variance and variance epsilon.

This public trainer is newly organized around the provider package. LoRA rank 16 / alpha 32, disabled dropout, microbatch generation, temperature 0.8 / top-p 0.95, weight decay 0, gradient norm 1, and clipping epsilon 0.2 are explicit implementation choices. They are not claimed to be verified historical paper settings. No published result table has been reproduced with this release.

The available 2026-08-07 historical archive includes `scripts/train_p0b_grpo.py`, using a local Qwen Judge, near-tie reward shaping, min_score=80, G=8, learning rate 5e-6, beta=0.02, and a DAPO loss default. It predates the supplied manuscript and does not implement its current reward protocol. That archive was inspected for provenance; its old scripts and internal cloud configuration are not advertised as a current reproduction.

Successful smoke training establishes that generation, scalar score parsing, group normalization, backpropagation, and adapter saving execute. It does not establish real-Judge policy adherence, preference transfer, independent carriers, model-level detection, or account attribution. Those need held-out behavioral data and an independently calibrated null pool.
