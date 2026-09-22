# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Opt-in GGUF + GenAI CPU acceptance checks against architecture_oracle.cpp output.

Use the GenAI build linked against the frontend under test on PYTHONPATH. The
reference contains an int32 vocabulary size followed by 13 F32 logit rows; the
companion .tokens file records identical reference histories. A successful load
alone never counts as a passing scenario. This is a bounded correctness check,
not model-quality or exhaustive generation coverage.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_genai as genai


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--backend", choices=["SDPA", "PA"], required=True)
    parser.add_argument("--mmproj", type=Path)
    parser.add_argument("--fresh-accuracy", action="store_true",
                        help="Diagnostic: rebuild the pipeline for each reference history")
    prefix = parser.add_mutually_exclusive_group()
    prefix.add_argument("--prefix-caching", dest="prefix_caching", action="store_true")
    prefix.add_argument("--no-prefix-caching", dest="prefix_caching", action="store_false")
    parser.set_defaults(prefix_caching=None)
    args = parser.parse_args()
    properties = dict(GGUF_READER="FRONTEND", ATTENTION_BACKEND=args.backend,
                      INFERENCE_PRECISION_HINT="f32", KV_CACHE_PRECISION="f16",
                      DYNAMIC_QUANTIZATION_GROUP_SIZE=0, INFERENCE_NUM_THREADS=4)
    if args.prefix_caching is not None:
        config = genai.SchedulerConfig()
        config.enable_prefix_caching = args.prefix_caching
        properties["scheduler_config"] = config
    report = dict(model=str(args.model.resolve()), backend=args.backend,
                  openvino_version=ov.get_version(), genai_version=genai.__version__,
                  fresh_accuracy=args.fresh_accuracy,
                  prefix_caching=(args.backend == "PA" if args.prefix_caching is None else args.prefix_caching),
                  scenarios={})

    def save():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")

    def make_pipeline():
        return genai.LLMPipeline(str(args.model), "CPU", **properties)

    pipeline = None

    def test(name, function):
        nonlocal pipeline
        try:
            if pipeline is None:
                pipeline = make_pipeline()
            report["scenarios"][name] = dict(status="pass", detail=function())
        except Exception as error:
            report["scenarios"][name] = dict(status="fail", error=str(error))
            # A failed chat or infer request must not contaminate subsequent scenarios.
            pipeline = None
        save()
        print(name, report["scenarios"][name]["status"], flush=True)

    def equal(actual, expected):
        assert actual == expected, (actual, expected)
        return actual

    vocab = int(np.fromfile(args.reference, np.int32, count=1)[0])
    logits = np.fromfile(args.reference, np.float32, offset=4).reshape(-1, vocab)
    expected = logits.argmax(-1).tolist()
    schedule = [list(map(int, line.split()))[1:]
                for line in Path(str(args.reference) + ".tokens").read_text().splitlines()]
    assert len(schedule) == len(expected) and len(schedule) > 0
    prompt = "The capital of France is"
    generation = dict(max_new_tokens=13, min_new_tokens=13, apply_chat_template=False)

    test("tokenizer", lambda: equal(pipeline.get_tokenizer().encode(prompt).input_ids.data.tolist()[0], schedule[0]))

    def accuracy():
        actual, history = [], []
        for ids in schedule:
            history.extend(ids)
            runner = make_pipeline() if args.fresh_accuracy else pipeline
            result = runner.generate(ov.Tensor(np.array([history], np.int64)), max_new_tokens=1, ignore_eos=True)
            actual.append(result.tokens[0][0])
        matches = sum(a == b for a, b in zip(actual, expected))
        metrics = dict(actual=actual, expected=expected, matching_choices=matches, total=len(expected))
        report["accuracy"] = metrics
        assert actual[0] == expected[0] and matches / len(expected) >= .9, metrics
        return metrics

    test("teacher_forced_accuracy", accuracy)
    test("greedy", lambda: str(pipeline.generate(prompt, **generation)))

    def chat():
        before = str(pipeline.generate(prompt, **generation))
        pipeline.start_chat()
        try:
            pipeline.generate("Hello", max_new_tokens=8)
            pipeline.generate("What did I just say?", max_new_tokens=8)
        finally:
            pipeline.finish_chat()
        return equal(str(pipeline.generate(prompt, **generation)), before)

    test("chat_and_reset", chat)

    def batch():
        prompts = [prompt, "2 + 2 ="]
        expected_text = [str(pipeline.generate(text, **generation)) for text in prompts]
        return equal(list(pipeline.generate(prompts, **generation).texts), expected_text)

    test("batch_two", batch)

    def beam():
        result = pipeline.generate(ov.Tensor(np.array([schedule[0]], np.int64)),
                                   max_new_tokens=8, min_new_tokens=8, num_beams=3, num_return_sequences=2)
        assert len(result.tokens) == 2 and all(len(tokens) <= 8 for tokens in result.tokens), result.tokens
        assert np.isfinite(result.scores).all()
        return dict(tokens=result.tokens, scores=result.scores)

    test("beam_search", beam)

    def streaming():
        chunks = []
        expected_text = str(pipeline.generate(prompt, **generation))
        pipeline.generate(prompt, streamer=lambda text: chunks.append(text), **generation)
        return equal("".join(chunks), expected_text)

    test("streaming", streaming)
    if args.mmproj:
        def multimodal_loading():
            genai.VLMPipeline(str(args.model), "CPU", mmproj_path=str(args.mmproj), **properties)
            return "Pair construction only; media generation requires separate acceptance coverage"
        test("multimodal_pair_loading", multimodal_loading)
    report["passed"] = all(value["status"] == "pass" for value in report["scenarios"].values())
    save()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
