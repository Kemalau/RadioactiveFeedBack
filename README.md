# RadioactiveFeedBack

Train a Student from scalar Judge feedback with your explicit private reward preferences.

[中文说明](README.zh-CN.md) · [Provider API package](https://github.com/Kemalau/OneLinetoProtectYourReward) · [Method and implementation](docs/METHOD.md)

The Student generates every candidate. The Judge evaluates the group and returns only final numerical scores. A GRPO update uses those scores to train the Student. The Judge never supplies reference answers, explanations, or chain of thought to the learner.

This release implements the pointwise protocol in the current Radioactive Feedback manuscript. It is a new, portable training entry point based on the provider prompt implementation, rather than the original scripts that produced historical checkpoints. It has not reproduced the paper's result tables.

## Install

Python 3.10 or newer. GPU training is recommended. Install PyTorch for your CUDA environment first, then:

```bash
git clone https://github.com/Kemalau/RadioactiveFeedBack.git
cd RadioactiveFeedBack
python -m pip install '.[train]'
```

For optional 4-bit LoRA training, install `'.[train,quantized]'` and use `--load-in-4bit`. The trainer uses one process and one Student device per run. The Judge runs behind your existing API.

## Prepare training tasks

A JSONL file needs only a unique task ID and a prompt:

```json
{"task_id":"code-001","prompt":"Write a Python function that ..."}
{"task_id":"math-001","prompt":"Solve the following problem: ..."}
```

An optional `carrier_id` fixes the carrier for a task. Without it, the private Judge prompt routes by task domain. Neither the carrier ID nor the key is passed in the Student's prompt. Extra input fields, including reference answers, are not sent to the Judge.

Provide a separate `--audit-file` to exclude overlapping task IDs and whitespace-normalized prompt texts. Internal duplicate training prompts are removed, then the retained prompts are shuffled once with the training seed. The final partial prompt batch is retained. `examples/train.jsonl` and `examples/audit.jsonl` are small synthetic examples, not the paper's dataset splits.

## Train with an existing Judge API

Configure a chat-completions endpoint:

```bash
export KEYFLIP_UPSTREAM_URL='https://your-judge.example/v1/chat/completions'
export KEYFLIP_UPSTREAM_MODEL='your-judge-model'
export KEYFLIP_UPSTREAM_API_KEY='your-api-credential'
```

Run:

```bash
rfeedback train \
  --student-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --student-revision 2e1fd397ee46e1388853d2af2c993145b0f1098a \
  --train-file ./examples/train.jsonl \
  --audit-file ./examples/audit.jsonl \
  --condition marked \
  --carrier-file ./private/my-carriers.json \
  --output-dir ./runs/marked-seed42
```

This command calls the Judge and updates model weights. The listed examples exercise the interface; they do not establish watermark transfer. Use `--dry-run` first to validate inputs and inspect the full plan without loading weights or calling the API. For a real run, specify your provider's model snapshot with `--judge-version` and an immutable Student revision.

By default the training process copies your preferred and opposite behavior definitions into the Judge's system prompt, using the same scoring policy as `OneLinetoProtectYourReward`. The Judge itself recognizes the behavior, forms the ordinary rubric score, and returns the final adjusted score in one call. If your Judge endpoint already has that policy installed, use `--judge-already-protected` to avoid applying it twice. The deployment owner must retain that endpoint's private configuration for auditing.

The client expects an OpenAI-style chat-completions envelope with content `{"scores": [95, 85, ...]}` in candidate order. Use `--no-json-mode` for providers that do not accept `response_format`. Malformed, missing, nonfinite, or out-of-range scores abort training; they are never silently replaced with zero.

## Private preferences and custom carriers

No concrete carrier definitions or registered preferences are bundled. Trainers supply their own private configuration, which directly states the behavior to reward and its opposite. This explicit preference set is the watermark configuration. There is no string seed, HMAC mapping, or separately derived direction. Only the Judge receives these rules; the Student receives ordinary task prompts and final numerical rewards.

Version 0.3 removes the `KEYFLIP_KEY` requirement and `--key-env`. Rewrite legacy `positive`/`negative` files as the intended `preferred`/`opposite` behaviors. Legacy files are rejected to avoid silently changing the preference. To reverse a preference, swap the two descriptions.

Copy the unfilled [carrier format template](examples/carriers.template.json) to a private location such as `private/my-carriers.json`:

```json
[
  {
    "id": "carrier_1",
    "domain": "",
    "preferred": "",
    "opposite": "",
    "abstain": ""
  }
]
```

Fill every empty field with your own task scope, preferred behavior, opposite behavior, and abstention rule. The Judge uses the preferred behavior for a positive adjustment and the opposite for a negative adjustment, subject to the quality floor. Both or neither, ambiguity, and inapplicability cause abstention. The blank template is intentionally rejected. Pass `--carrier-file ./private/my-carriers.json` whenever injecting a policy. Omit `--k` to use the complete pool, or choose any k up to the number of defined carriers. `--carriers` selects and orders IDs; `--carrier` fixes one selected carrier for all tasks. Automatic routing selects the first matching task domain, so separate overlapping task scopes or use per-task `carrier_id` assignments. Increasing k does not increase the shift per answer.

The `private/` directory and `my-carriers.json` are ignored by Git. Keep your filled configuration and generated prompt under your own control; neither needs to be committed to the public repository.

## Conditions

| Condition | Feedback |
| --- | --- |
| `clean` | Ordinary Judge scores with no private preference policy injected; use an unmarked endpoint |
| `marked` | Registered carrier preferences in the Judge's private prompt |
| `sham` | Preferences over a separate decoy carrier file |

For Sham, pass both `--carrier-file ./decoys.json` and `--registered-carrier-file ./registered.json`. The CLI rejects overlapping IDs. ID separation alone cannot establish that the behavioral features are independent; match the decoys' exposure, task coverage, and strength in the experimental design. Base is the initial Student evaluated without training.

Keep the seed, input tasks, revisions, rollout budget, and optimizer settings matched across comparison conditions. The same seed fixes the initial sampling RNG and task ordering; after policies diverge, it does not imply identical generated answers.

## Defaults and reward construction

| Setting | Default |
| --- | --- |
| Pass over retained training tasks | One shuffled epoch, no replacement |
| Tasks per optimizer update | 50; final partial batch retained |
| Responses per task | 16 |
| Optimizer / learning rate | AdamW / `1e-6` |
| Frozen-reference KL coefficient | `0.04` |
| Judge scale / quality floor | 0–100 / 60 |
| Keyed shift | `rho=0.05`, at most ±5 points |
| Maximum generated tokens | 2048 |
| Student sampling | temperature 0.8 / top-p 0.95 |
| Adapter | LoRA rank 16 / alpha 32; dropout disabled |

Each task uses exactly one carrier. In the same call, the Judge internally forms the ordinary rubric score `r0`, classifies preference agreement `s`, and returns:

```text
s = +1 for preferred, -1 for opposite, 0 for abstention
eligible = (r0 >= 60) and (s in {-1, +1})
reward = clip(r0 + rho * R * eligible * s, 0, R)
```

Here `s` represents the manuscript's `c_j * phi_j`, with the preferred side directly specified by the operator. Appendix B states that the paper's experiments register the code explicitly. Eligibility is independent of other candidates and their score differences. The learner receives final rewards only, not `r0`, `s`, or the internal adjustment. This is a prompt instruction, not server-side numerical enforcement. The provider model's adherence and the resulting Student transfer need separate evaluation.

See [docs/METHOD.md](docs/METHOD.md) for the exact group advantage, clipped objective, KL estimate, and implementation choices that are not specified in the manuscript.

## Outputs

Every run requires a fresh output directory:

```text
runs/marked-seed42/
  manifest.json          # versions, protocol hash, counts, state
  trajectory.jsonl       # retained task sequence, once per task
  reward_receipts.jsonl  # final scores, request hashes, parse success
  rollouts.jsonl         # Student answers and final rewards
  metrics.jsonl          # update loss, KL, gradient norm, counts
  checkpoint-10/         # optional intermediate model/adapter snapshot
  adapter/               # terminal LoRA adapter and tokenizer
```

LoRA weights must change and all expected rollouts must complete before the manifest is marked complete. `--lora-rank 0` saves a full model and uses a separate frozen reference model. Interrupted runs preserve their failure state and partial output; automatic optimizer resume is not implemented in this release. Run directories are created with mode `0700`; their contents are private experimental artifacts.

## Verification

```bash
python -m unittest discover -s tests -v
python tests/smoke_train.py
```

The smoke test creates a randomly initialized tiny Student locally and uses a mock scalar Judge with explicit private preferences. It performs two GRPO updates, includes a partial final prompt batch, and verifies a nonzero saved LoRA adapter and the recorded preference configuration. It makes no remote model calls. It verifies the training path, not watermark effectiveness or paper results.

This repository does not include production credentials, private keys, training corpora, experiment checkpoints, cloud submission infrastructure, or a model-level detector. Use a separate held-out audit and null calibration to measure transfer and false-positive rates.

## License

[MIT](LICENSE). External models and datasets retain their own licenses.
