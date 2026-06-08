# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""vLLM-specific glue for the OpenVINO torchdynamo backend.

Modules:
    plugin           - vLLM general_plugins entry point; wires the OV backend
                       into vllm.v1.worker.cpu_model_runner.
    paged_attention  - Custom torch op + FX rewrite that converts vLLMs
                       unified_attention_with_output HOP into a flat OV PA op.
    sampler          - OV-compiled fast path for vllm.v1.sample.sampler.Sampler,
                       gated by an eligibility check.
"""
