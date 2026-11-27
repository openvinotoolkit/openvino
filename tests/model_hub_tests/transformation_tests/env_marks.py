# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Transformation-test models that cannot run with pytorch/envs/transformations_actual.txt.
# They are marked `transformations_legacy`; job_pytorch_models_tests.yml installs
# pytorch/envs/transformations_legacy.txt and runs them in a separate step.
import pytest

transformations_legacy = pytest.mark.transformations_legacy

PA_LEGACY_MODELS = {
    # optimum-intel caps these architectures at transformers<=4.53.3
    "optimum-intel-internal-testing/tiny-random-deepseek-v3",
    "optimum-intel-internal-testing/tiny-random-minicpm",
    "optimum-intel-internal-testing/tiny-random-minicpm3",
    "optimum-intel-internal-testing/tiny-random-snowflake",
    "optimum-intel-internal-testing/tiny-random-nanollava",
    "optimum-intel-internal-testing/tiny-random-phi-4-multimodal",
    "optimum-intel-internal-testing/tiny-random-phi3-vision",
    # optimum-intel caps chatglm and qwen at transformers<=4.55.4
    "optimum-intel-internal-testing/tiny-random-chatglm",
    "optimum-intel-internal-testing/tiny-random-chatglm4",
    "optimum-intel-internal-testing/tiny-random-qwen",
    # auto-gptq needs optimum 1.x, optimum 2.x loads GPTQ models only through gptqmodel
    "optimum-intel-internal-testing/opt-125m-gptq-4bit",
    # autoawq does not support transformers>=4.57
    "optimum-intel-internal-testing/tiny-mixtral-AWQ-4bit",
}

ROPE_LEGACY_MODELS = {
    # optimum-intel caps minicpm at transformers<=4.53.3, chatglm and qwen at <=4.55.4
    "katuni4ka/tiny-random-minicpm",
    "katuni4ka/tiny-random-qwen",
    "optimum-intel-internal-testing/tiny-random-chatglm",
    "optimum-intel-internal-testing/tiny-random-chatglm4",
}


def with_env_marks(cases, legacy_models, name_index, unpack):
    missing = legacy_models - {case[name_index] for case in cases}
    assert not missing, f"Legacy models are missing from the model list: {sorted(missing)}"
    return [pytest.param(*(case if unpack else (case,)),
                         marks=transformations_legacy if case[name_index] in legacy_models else ())
            for case in cases]
