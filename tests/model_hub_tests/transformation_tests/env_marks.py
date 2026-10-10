# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Transformation-test models that cannot run with pytorch/envs/optimum_intel.txt.
# Each marker is named after the pytorch/envs/<marker>.txt it needs; job_pytorch_models_tests.yml
# installs that file and runs the marked tests in a separate step.
import os

import pytest

# optimum-intel caps these architectures at transformers<=4.53.3
optimum_intel_legacy = pytest.mark.optimum_intel_legacy
# optimum 2.x supports GPTQ only through gptqmodel, which requires transformers 5.x
optimum_intel_gptq = pytest.mark.optimum_intel_gptq

PA_MODEL_MARKS = {
    "optimum-intel-internal-testing/opt-125m-gptq-4bit": optimum_intel_gptq,
    "optimum-intel-internal-testing/tiny-random-deepseek-v3": optimum_intel_legacy,
    "optimum-intel-internal-testing/tiny-random-minicpm": optimum_intel_legacy,
    "optimum-intel-internal-testing/tiny-random-minicpm3": optimum_intel_legacy,
    "optimum-intel-internal-testing/tiny-random-snowflake": optimum_intel_legacy,
    "optimum-intel-internal-testing/tiny-random-nanollava": optimum_intel_legacy,
    "optimum-intel-internal-testing/tiny-random-phi-4-multimodal": optimum_intel_legacy,
    "optimum-intel-internal-testing/tiny-random-phi3-vision": optimum_intel_legacy,
}

ROPE_MODEL_MARKS = {
    "katuni4ka/tiny-random-minicpm": optimum_intel_legacy,
}


def with_env_marks(cases, model_marks, name_index, unpack):
    names = {case[name_index] for case in cases}
    missing = set(model_marks) - names
    assert not missing, f"Models with env marks are missing from the model list: {sorted(missing)}"
    envs_dir = os.path.join(os.path.dirname(__file__), "..", "pytorch", "envs")
    for name in {mark.name for mark in model_marks.values()}:
        assert os.path.isfile(os.path.join(envs_dir, name + ".txt")), f"No pytorch/envs/{name}.txt for marker {name}"
    return [pytest.param(*(case if unpack else (case,)), marks=model_marks.get(case[name_index], ()))
            for case in cases]
