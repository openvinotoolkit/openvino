#!/usr/bin/env python3

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Test inference with conditional compiled binaries.
"""

from tests_utils import run_infer


def test_infer(test_id, model, artifacts):
    """ Test inference with conditional compiled binaries
    """
    install_prefix = artifacts / test_id / "install_pkg"
    out = artifacts / test_id
    returncode, output = run_infer(model, f"{out}_cc.npz", install_prefix)
    assert returncode == 0, f"Command exited with non-zero status {returncode}:\n {output}"
