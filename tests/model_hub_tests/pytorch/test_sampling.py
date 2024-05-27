# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common import (
    get_models_list,
    run_pa
)

@pytest.mark.precommit
@pytest.mark.parametrize("model_id", get_models_list("pytorch/models/precommit"))
def test_pa_precommit(tmp_path, model_id, ie_device):
    run_pa(tmp_path, model_id)

@pytest.mark.nightly
@pytest.mark.parametrize("model_id", get_models_list("pytorch/models/nightly"))
def test_pa_nightly(tmp_path, model_id, ie_device):
    run_pa(tmp_path, model_id)
