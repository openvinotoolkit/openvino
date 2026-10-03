#!/usr/bin/env python3
"""lm-evaluation-harness on the OpenVINO GPU plugin (TernOCL int2) through
paged_serve_llm_27b: every generate_until batch is written as token ids to a
request file, served with continuous batching, and read back.

    python lm_eval_ov.py --lm <u2 dir>/openvino_model.xml \
        --embed <u2 dir>/openvino_text_embeddings_model.xml \
        --tokenizer <HF dir with tokenizer + chat template> \
        --serve <build>/paged_serve_llm_27b --tasks gsm8k_cot_llama \
        [--limit N] [--batch 8] [--think medium|xhigh|off] [--max-gen 4096] \
        [--gpus 0,1,...] [--metadata '{"max_seq_lengths": [8192]}'] [--out <dir>]

Greedy, chat template applied (few-shot as multi-turn), thinking on by default
with the answer taken after </think>, i.e. the same protocol as
scripts/eval_bonsai2_lm_eval.sh in the vLLM plugin.
"""
import argparse
import json
import os
import subprocess
import sys
import time

from lm_eval import simple_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.utils import make_table


@register_model("ov_paged_serve")
class OVPagedServe(LM):
    def __init__(self, lm_xml, embed_xml, tokenizer, serve, batch=8, think="medium",
                 max_gen=4096, max_len=6144, workdir=".", device="GPU", gpus=None, **kw):
        super().__init__()
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tokenizer)
        self.lm_xml, self.embed_xml, self.serve = lm_xml, embed_xml, serve
        self.batch, self.think, self.max_gen, self.max_len = batch, think, max_gen, max_len
        self.workdir, self.ov_device = workdir, device
        self.gpus = gpus or [None]
        self.eos = [248044, 248046]
        self.calls = 0
        self.total_gen_tokens = 0
        self.serve_seconds = 0.0

    # -- chat template -----------------------------------------------------
    @property
    def tokenizer_name(self):
        return "bonsai2-27b"

    @property
    def chat_template(self):
        return self.tok.chat_template

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        kw = {}
        if self.think == "off":
            kw["enable_thinking"] = False
        else:
            kw["enable_thinking"] = True
            kw["reasoning_effort"] = self.think
        return self.tok.apply_chat_template(chat_history, tokenize=False,
                                            add_generation_prompt=add_generation_prompt,
                                            continue_final_message=not add_generation_prompt, **kw)

    def tok_encode(self, s, **kw):
        return self.tok.encode(s, add_special_tokens=False)

    # -- generation --------------------------------------------------------
    def generate_until(self, requests, disable_tqdm=False):
        ctxs = [r.args[0] for r in requests]
        gks = [r.args[1] for r in requests]
        ids = [self.tok_encode(c) for c in ctxs]
        # Round-robin shards, one serve process per GPU; outputs are merged back in request order.
        shards = [list(range(s, len(ids), len(self.gpus))) for s in range(len(self.gpus))]
        procs = []
        t0 = time.perf_counter()
        print(f"[ov-serve] {len(requests)} requests on {len(self.gpus)} GPU(s), prompt tokens "
              f"min/median/max {min(map(len, ids))}/{sorted(map(len, ids))[len(ids)//2]}/{max(map(len, ids))}",
              file=sys.stderr, flush=True)
        for s, (gpu, shard) in enumerate(zip(self.gpus, shards)):
            if not shard:
                continue
            tag = f"{self.calls}" if len(self.gpus) == 1 else f"{self.calls}.{s}"
            req_path = os.path.join(self.workdir, f"requests_{tag}.txt")
            out_path = os.path.join(self.workdir, f"outputs_{tag}.txt")
            with open(req_path, "w") as fh:
                for j in shard:
                    g = gks[j]
                    mx = int(g.get("max_gen_toks", self.max_gen) or self.max_gen)
                    mx = min(mx, self.max_len - len(ids[j]))
                    fh.write(f"{mx};{','.join(map(str, ids[j]))}\n")
            env = dict(os.environ)
            env.setdefault("BENCH_MAX_LEN", str(self.max_len))
            env.setdefault("BENCH_PRECISION", "f16")
            env.setdefault("BENCH_VERBOSE", "1")
            if gpu is not None:
                env["ZE_AFFINITY_MASK"] = gpu
                if "BENCH_TIMINGS" in env:
                    env["BENCH_TIMINGS"] += f".{s}"
            cmd = [self.serve, self.lm_xml, self.embed_xml, self.ov_device, req_path, out_path, str(self.batch)]
            procs.append((subprocess.Popen(cmd, env=env), shard, out_path))
        lines = [None] * len(requests)
        for proc, shard, out_path in procs:
            if proc.wait() != 0:
                for other, _, _ in procs:
                    other.kill()
                raise subprocess.CalledProcessError(proc.returncode, proc.args)
            with open(out_path) as fh:
                shard_lines = fh.read().splitlines()
            assert len(shard_lines) == len(shard), (out_path, len(shard_lines), len(shard))
            for j, line in zip(shard, shard_lines):
                lines[j] = line
        self.serve_seconds += time.perf_counter() - t0
        self.calls += 1
        res = []
        for line, g in zip(lines, gks):
            out_ids = [int(x) for x in line.split(",") if x]
            self.total_gen_tokens += len(out_ids)
            if out_ids and out_ids[-1] in self.eos:
                out_ids = out_ids[:-1]
            text = self.tok.decode(out_ids, skip_special_tokens=False)
            if self.think != "off" and "</think>" in text:
                text = text.split("</think>")[-1]
            for stop in g.get("until", []) or []:
                if stop and stop in text:
                    text = text.split(stop)[0]
            res.append(text)
        return res

    def loglikelihood(self, requests, **kw):
        raise NotImplementedError("generative tasks only")

    def loglikelihood_rolling(self, requests, **kw):
        raise NotImplementedError("generative tasks only")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", required=True)
    ap.add_argument("--embed", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--serve", required=True)
    ap.add_argument("--tasks", default="gsm8k_cot_llama")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--think", default="medium", help="medium|xhigh|off")
    ap.add_argument("--max-gen", type=int, default=4096)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--gpus", default=None,
                    help="comma list of GPU indices (ZE_AFFINITY_MASK); requests are split across them")
    ap.add_argument("--metadata", default="{}",
                    help="JSON task metadata, e.g. RULER max_seq_lengths; the tokenizer is added")
    ap.add_argument("--out", default="lm_eval_ov_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    lm = OVPagedServe(a.lm, a.embed, a.tokenizer, a.serve, batch=a.batch, think=a.think,
                      max_gen=a.max_gen, max_len=a.max_len, workdir=a.out, device=a.device,
                      gpus=a.gpus.split(",") if a.gpus else None)
    t0 = time.perf_counter()
    results = simple_evaluate(model=lm, tasks=a.tasks.split(","), limit=a.limit,
                              apply_chat_template=True, fewshot_as_multiturn=True,
                              gen_kwargs={"temperature": 0, "max_gen_toks": a.max_gen},
                              log_samples=True, confirm_run_unsafe_code=True,
                              metadata={"tokenizer": a.tokenizer} | json.loads(a.metadata))
    wall = time.perf_counter() - t0
    print(make_table(results))
    print(f"wall {wall:.0f} s, serve {lm.serve_seconds:.0f} s, generated tokens {lm.total_gen_tokens}")
    samples = results.pop("samples", {})
    with open(os.path.join(a.out, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=str)
    for task, rows in samples.items():
        with open(os.path.join(a.out, f"samples_{task}.jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=str) + "\n")


if __name__ == "__main__":
    main()
