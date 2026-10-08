from __future__ import annotations

from contextlib import nullcontext
import copy
from dataclasses import asdict, dataclass
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import time

from .data import digest_json, prepare_tasks, read_tasks, task_batches, write_jsonl
from .judge import JudgeClient


STUDENT_SYSTEM = "Solve the submitted task accurately. Give your answer and the reasoning needed to justify it."


@dataclass(frozen=True)
class TrainConfig:
    student_model: str
    student_revision: str
    train_file: str
    output_dir: str
    judge_version: str = "unspecified"
    audit_file: str | None = None
    condition: str = "marked"
    seed: int = 42
    prompt_batch_size: int = 50
    num_generations: int = 16
    learning_rate: float = 1e-6
    beta: float = 0.04
    epsilon: float = 0.2
    max_prompt_tokens: int = 2048
    max_new_tokens: int = 2048
    temperature: float = 0.8
    top_p: float = 0.95
    generation_batch_size: int = 4
    lora_rank: int = 16
    lora_alpha: int = 32
    load_in_4bit: bool = False
    device: str = "auto"
    student_system: str = STUDENT_SYSTEM
    save_every: int = 10
    max_grad_norm: float = 1.0

    def __post_init__(self):
        for name in ("student_model", "student_revision", "train_file", "output_dir", "student_system"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        if self.condition not in {"clean", "marked", "sham"}:
            raise ValueError("unknown training condition")
        if not Path(self.student_model).is_dir() and not re.fullmatch(r"[0-9a-fA-F]{40}", self.student_revision):
            raise ValueError("remote Student revision must be an immutable 40-character commit hash")
        for name in ("prompt_batch_size", "num_generations", "max_prompt_tokens", "max_new_tokens",
                     "generation_batch_size", "lora_alpha", "save_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_generations < 2 or self.lora_rank < 0:
            raise ValueError("GRPO requires at least two responses; LoRA rank must be nonnegative")
        for name in ("learning_rate", "beta", "epsilon", "temperature", "top_p", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate <= 0 or self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("learning rate, temperature, and top_p must be positive")
        if self.beta < 0 or self.epsilon < 0 or self.max_grad_norm <= 0:
            raise ValueError("invalid KL, clipping, or gradient norm parameter")


def prepare_run(config: TrainConfig) -> tuple[list[dict], dict]:
    source = read_tasks(config.train_file)
    audit = read_tasks(config.audit_file) if config.audit_file else []
    tasks, data = prepare_tasks(source, audit, config.seed)
    metadata = {"config": asdict(config), "data": data,
        "updates": math.ceil(len(tasks) / config.prompt_batch_size),
        "expected_rollouts": len(tasks) * config.num_generations,
        "epochs": 1, "reward_path": "one Judge call per task group; final scalar scores only",
        "advantage": "(reward - group_mean) / sqrt(sample_variance + 1e-6)",
        "loss": "clipped token-level GRPO; mean completion tokens per response, then mean responses",
        "reference": "frozen initial Student; base model with disabled zero-initialized LoRA adapter",
        "historical_checkpoint_reproduction": False}
    if config.lora_rank == 0:
        metadata["reference"] = "deep copy of frozen initial Student"
    return tasks, metadata


def private_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def completion_mask(completions, eos_ids, pad_id):
    import torch
    mask = torch.ones_like(completions, dtype=torch.bool)
    for row in range(completions.shape[0]):
        for index, value in enumerate(completions[row].tolist()):
            if value in eos_ids:
                mask[row, index + 1:] = False
                break
            if pad_id is not None and value == pad_id:
                mask[row, index:] = False
                break
    if torch.any(mask.sum(dim=1) == 0):
        raise RuntimeError("Student generated an empty completion")
    return mask


def token_logps(model, input_ids, attention_mask, prompt_length):
    import torch
    # Chunked cross-entropy avoids an additional full-vocabulary log_softmax tensor.
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    logits = logits[:, prompt_length - 1:-1, :].float()
    labels = input_ids[:, prompt_length:]
    return -torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
    ).reshape(labels.shape)


def train(config: TrainConfig, judge: JudgeClient) -> dict:
    tasks, manifest = prepare_run(config)
    # Validate task-assigned carriers before importing the training stack or loading weights.
    policy_hashes = sorted({digest_json(judge.system_prompt(task)) for task in tasks})
    root = Path(config.output_dir)
    if root.exists():
        raise ValueError("output directory already exists; choose a fresh run namespace")
    root.mkdir(parents=True)
    os.chmod(root, 0o700)
    manifest["judge"] = {"model": judge.model, "version": config.judge_version,
        "endpoint_sha256": digest_json(judge.url), "score_max": judge.score_max,
        "system_prompt_sha256": policy_hashes, "policy_injected": judge.policy is not None}
    if judge.policy is not None:
        policy = judge.policy
        manifest["provider_policy"] = {"version": "pointwise-v2", "key_mapping": "keyflip-api/prompt-v1",
            "rho": policy.rho, "score_max": policy.score_max, "min_score": policy.min_score,
            "k": policy.k, "assigned_carrier": policy.assigned_carrier,
            "carriers": [{"id": carrier.identifier, "domain": carrier.domain, "rule": carrier.rule,
                "direction": policy.direction(carrier)} for carrier in policy.selected]}
    manifest["protocol_sha256"] = digest_json(manifest)
    manifest["state"] = "starting"
    private_json(root / "manifest.json", manifest)
    write_jsonl(root / "trajectory.jsonl", tasks)
    try:
        result = _train(config, judge, tasks, root, manifest)
    except BaseException as error:
        manifest["state"] = "failed"
        manifest["failure_type"] = type(error).__name__
        private_json(root / "manifest.json", manifest)
        raise
    manifest["state"] = "complete"
    manifest["result"] = result
    private_json(root / "manifest.json", manifest)
    return result


def _train(config, judge, tasks, root, manifest):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from .objective import group_advantages, grpo_loss

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("this release supports one process and one Student device per run")
    set_seed(config.seed)
    device = config.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if config.load_in_4bit and (not device.startswith("cuda") or config.lora_rank == 0):
        raise ValueError("4-bit training requires a CUDA device and LoRA")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(config.student_model, revision=config.student_revision)
    if not tokenizer.chat_template:
        raise ValueError("Student tokenizer requires a chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        raise ValueError("Student tokenizer needs a pad or EOS token")
    tokenizer.padding_side = "left"
    kwargs = {"revision": config.student_revision, "dtype": dtype, "attn_implementation": "sdpa"}
    if config.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(config.student_model, **kwargs)
    if not config.load_in_4bit:
        model.to(device)
    base_revision = getattr(model.config, "_commit_hash", None) or config.student_revision
    manifest["resolved_student_revision"] = base_revision
    reference_model = None
    if config.lora_rank:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if config.load_in_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False})
        model = get_peft_model(model, LoraConfig(r=config.lora_rank, lora_alpha=config.lora_alpha,
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules="all-linear"))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    else:
        reference_model = copy.deepcopy(model).eval()
        reference_model.requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.config.use_cache = False
    model.eval()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=0.0)
    manifest["versions"] = {name: importlib.metadata.version(name) for name in
                            ("torch", "transformers", "accelerate", "peft")}
    manifest["state"] = "training"
    private_json(root / "manifest.json", manifest)
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos] if eos is not None else [])
    total_rollouts = 0
    last_loss = None
    started = time.monotonic()

    for step, batch in enumerate(task_batches(tasks, config.prompt_batch_size), start=1):
        optimizer.zero_grad(set_to_none=True)
        rollouts = []
        score_rows = []
        for task in batch:
            messages = [{"role": "system", "content": config.student_system},
                        {"role": "user", "content": task["prompt"]}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            if inputs.input_ids.shape[1] > config.max_prompt_tokens:
                raise ValueError("Student prompt exceeds max_prompt_tokens; no silent truncation is performed")
            inputs = {name: value.to(device) for name, value in inputs.items()}
            prompt_length = inputs["input_ids"].shape[1]
            task_rollouts = []
            candidates = []
            for start in range(0, config.num_generations, config.generation_batch_size):
                count = min(config.generation_batch_size, config.num_generations - start)
                with torch.no_grad():
                    generated = model.generate(**inputs, do_sample=True, temperature=config.temperature,
                        top_p=config.top_p, num_return_sequences=count, max_new_tokens=config.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id, use_cache=True,
                        suppress_tokens=[tokenizer.pad_token_id] if tokenizer.pad_token_id not in eos_ids else None)
                completions = generated[:, prompt_length:]
                mask = completion_mask(completions, eos_ids, tokenizer.pad_token_id)
                for row in range(count):
                    length = int(mask[row].sum().item())
                    ids = generated[row:row + 1, :prompt_length + length]
                    completion_tokens = ids[0, prompt_length:]
                    candidates.append(tokenizer.decode(completion_tokens, skip_special_tokens=True))
                    task_rollouts.append({"ids": ids.cpu(), "prompt_length": prompt_length,
                                          "mask": torch.ones((1, length), dtype=torch.bool)})
            scores = judge.score_group(task, candidates)
            score_rows.append(scores)
            write_jsonl_append(root / "reward_receipts.jsonl", {"step": step, **judge.last_receipt})
            for index, (row, candidate, score) in enumerate(zip(task_rollouts, candidates, scores)):
                write_jsonl_append(root / "rollouts.jsonl", {"step": step, "task_id": task["task_id"],
                    "sample_id": index, "answer": candidate, "reward": score})
                rollouts.append(row)
        advantages = group_advantages(torch.tensor(score_rows, dtype=torch.float32)).flatten()
        batch_loss = batch_kl = 0.0
        # Cache old-policy and frozen-reference log probabilities before any update.
        with torch.no_grad():
            for row in rollouts:
                ids = row["ids"].to(device)
                attention = torch.ones_like(ids)
                row["old"] = token_logps(model, ids, attention, row["prompt_length"]).cpu()
                context = model.disable_adapter() if config.lora_rank else nullcontext()
                with context:
                    reference = model if config.lora_rank else reference_model
                    row["reference"] = token_logps(reference, ids, attention, row["prompt_length"]).cpu()
        model.train()
        # Accumulate one response at a time; the optimizer steps once per prompt batch.
        for index, row in enumerate(rollouts):
            ids = row["ids"].to(device)
            current = token_logps(model, ids, torch.ones_like(ids), row["prompt_length"])
            loss, diagnostics = grpo_loss(current, row["old"].to(device), row["reference"].to(device),
                row["mask"].to(device), advantages[index:index + 1].to(device),
                beta=config.beta, epsilon=config.epsilon)
            (loss / len(rollouts)).backward()
            batch_loss += loss.detach().item() / len(rollouts)
            batch_kl += diagnostics["kl"] / len(rollouts)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("nonfinite gradient norm; update rejected")
        optimizer.step()
        model.eval()
        total_rollouts += len(rollouts)
        last_loss = batch_loss
        record = {"step": step, "prompts": len(batch), "rollouts": len(rollouts), "loss": batch_loss,
            "kl": batch_kl, "gradient_norm": grad_norm.item(),
            "reward_mean": float(torch.tensor(score_rows).mean()),
            "total_rollouts": total_rollouts, "elapsed_seconds": time.monotonic() - started}
        write_jsonl_append(root / "metrics.jsonl", record)
        print(json.dumps(record), flush=True)
        if step % config.save_every == 0:
            save_model(model, tokenizer, root / f"checkpoint-{step}")
    final = root / ("adapter" if config.lora_rank else "model")
    save_model(model, tokenizer, final)
    changed = any(torch.count_nonzero(value).item() for name, value in model.named_parameters() if "lora_B" in name)
    if config.lora_rank and not changed:
        raise RuntimeError("LoRA output weights stayed zero; no trained adapter was produced")
    if total_rollouts != manifest["expected_rollouts"]:
        raise RuntimeError("rollout count does not match frozen training plan")
    return {"updates": manifest["updates"], "rollouts": total_rollouts,
        "final_loss": last_loss, "final_checkpoint": final.name, "nonzero_lora": bool(changed)}


def write_jsonl_append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_model(model, tokenizer, path: Path):
    path.mkdir()
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
