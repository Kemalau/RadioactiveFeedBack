from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .judge import JudgeClient, RUBRIC
from .provider import PromptConfig, build_system_prompt, load_carriers
from .training import STUDENT_SYSTEM, TrainConfig, prepare_run, train


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train a Student with GRPO and scalar Judge feedback")
    sub = parser.add_subparsers(dest="command", required=True)
    prompt = sub.add_parser("prompt", help="generate a provider-side private scoring prompt")
    run = sub.add_parser("train", help="run one epoch, or inspect a plan with --dry-run")
    for command in (prompt, run):
        command.add_argument("--carrier-file", type=Path, help="your private carrier definitions; no carriers are bundled")
        command.add_argument("--carriers", help="comma-separated carrier IDs, in routing order")
        command.add_argument("--k", type=int, help="number of carriers; default all selected")
        command.add_argument("--carrier", help="fix one selected carrier for the endpoint")
        command.add_argument("--rho", type=float, default=0.05)
        command.add_argument("--score-max", type=float, default=100)
        command.add_argument("--min-score", type=float, default=60)
        command.add_argument("--judge-system", type=Path, help="custom Judge grading rubric")
    prompt.add_argument("--output", type=Path, required=True)
    run.add_argument("--student-model", required=True)
    run.add_argument("--student-revision", required=True, help="immutable model commit or a local snapshot revision")
    run.add_argument("--train-file", required=True)
    run.add_argument("--audit-file", help="held-out tasks to remove from training")
    run.add_argument("--output-dir", required=True)
    run.add_argument("--condition", choices=("clean", "marked", "sham"), default="marked")
    run.add_argument("--judge-url", default=os.environ.get("KEYFLIP_UPSTREAM_URL"))
    run.add_argument("--judge-model", default=os.environ.get("KEYFLIP_UPSTREAM_MODEL"))
    run.add_argument("--judge-version", default="unspecified", help="provider model snapshot or deployment revision for the manifest")
    run.add_argument("--judge-api-key-env", default="KEYFLIP_UPSTREAM_API_KEY")
    run.add_argument("--judge-already-protected", action="store_true",
                     help="use scoring rules already installed in the upstream API; do not inject twice")
    run.add_argument("--registered-carrier-file", type=Path,
                     help="enrolled carrier definitions; required for disjoint-ID checking of Sham")
    run.add_argument("--no-json-mode", action="store_true", help="omit API response_format for providers that do not support it")
    run.add_argument("--judge-timeout", type=float, default=120)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--prompt-batch-size", type=int, default=50)
    run.add_argument("--num-generations", type=int, default=16)
    run.add_argument("--learning-rate", type=float, default=1e-6)
    run.add_argument("--beta", type=float, default=0.04)
    run.add_argument("--epsilon", type=float, default=0.2)
    run.add_argument("--max-prompt-tokens", type=int, default=2048)
    run.add_argument("--max-new-tokens", type=int, default=2048)
    run.add_argument("--temperature", type=float, default=0.8)
    run.add_argument("--top-p", type=float, default=0.95)
    run.add_argument("--generation-batch-size", type=int, default=4)
    run.add_argument("--lora-rank", type=int, default=16, help="0 for full-weight training")
    run.add_argument("--lora-alpha", type=int, default=32)
    run.add_argument("--load-in-4bit", action="store_true")
    run.add_argument("--device", default="auto")
    run.add_argument("--student-system", type=Path)
    run.add_argument("--save-every", type=int, default=10)
    run.add_argument("--max-grad-norm", type=float, default=1.0)
    run.add_argument("--dry-run", action="store_true", help="validate and show plan without model load or API calls")
    args = parser.parse_args(argv)
    try:
        if args.score_max <= 0:
            raise ValueError("score_max must be positive")
        rubric = args.judge_system.read_text(encoding="utf-8") if args.judge_system else RUBRIC.replace("0 to 100", f"0 to {args.score_max:g}")
        if not rubric.strip():
            raise ValueError("Judge rubric is empty")
        pool = load_carriers(args.carrier_file) if args.carrier_file else ()
        ids = tuple(value.strip() for value in args.carriers.split(",")) if args.carriers else tuple(c.identifier for c in pool)
        policy = None
        use_policy = args.command == "prompt" or (args.condition != "clean" and not args.judge_already_protected)
        if use_policy:
            if not args.carrier_file:
                raise ValueError("provide --carrier-file with your private rules; no carriers are bundled")
            policy = PromptConfig(
                k=args.k if args.k is not None else len(ids), rho=args.rho,
                score_max=args.score_max, min_score=args.min_score, default_prompt=rubric,
                carriers=ids, assigned_carrier=args.carrier, carrier_pool=pool)
        if args.command == "prompt":
            fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(build_system_prompt(policy) + "\n")
            return 0
        if args.condition == "clean" and args.judge_already_protected:
            raise ValueError("Clean requires an unmarked Judge endpoint")
        if args.condition == "sham":
            if not args.carrier_file or not args.registered_carrier_file:
                raise ValueError("Sham requires a decoy --carrier-file and a --registered-carrier-file")
            registered = load_carriers(args.registered_carrier_file)
            if set(ids) & {c.identifier for c in registered}:
                raise ValueError("Sham decoy carrier IDs overlap enrolled carrier IDs")
        if not args.judge_url or not args.judge_model:
            raise ValueError("configure judge-url and judge-model, or KEYFLIP_UPSTREAM_URL and KEYFLIP_UPSTREAM_MODEL")
        judge = JudgeClient(url=args.judge_url, model=args.judge_model,
            api_key=os.environ.get(args.judge_api_key_env), policy=policy, rubric=rubric,
            timeout=args.judge_timeout, score_max=args.score_max, json_mode=not args.no_json_mode)
        config = TrainConfig(**{name: getattr(args, name) for name in TrainConfig.__dataclass_fields__
                                if name != "student_system"},
            student_system=args.student_system.read_text(encoding="utf-8") if args.student_system else STUDENT_SYSTEM)
        if args.dry_run:
            tasks, plan = prepare_run(config)
            for task in tasks:
                judge.system_prompt(task)
            plan["judge"] = {"model": judge.model, "policy_injected": policy is not None,
                "already_protected": args.judge_already_protected,
                "rho": args.rho, "score_max": args.score_max, "min_score": args.min_score,
                "k": policy.k if policy else None}
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        result = train(config, judge)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
