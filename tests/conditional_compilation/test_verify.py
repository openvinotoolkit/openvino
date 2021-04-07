#!/usr/bin/env python3
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Test to verify inference results.
"""

import sys
import glob
from inspect import getsourcefile
from pathlib import Path
import numpy as np
from proc_utils import cmd_exec  # pylint: disable=import-error
from install_pkg import get_openvino_environment  # pylint: disable=import-error


def run_infer(model, out, install_dir):
    """ Function running inference
    """

    cmd_exec(
        [sys.executable,
         str((Path(getsourcefile(lambda: 0)) / ".." / "tools" / "infer_tool.py").resolve()),
         "-d=CPU", f"-m={model}", f"-r={out}"
         ],
        env=get_openvino_environment(install_dir),
    )


def test_verify(test_id, model, artifacts, openvino_root_dir, openvino_cc, tolerance=0.000001):  # pylint: disable=too-many-arguments
    """ Test verifying that inference results are equal
    """
    out = artifacts / test_id
    run_infer(model, out, openvino_root_dir)
    run_infer(model, f"{out}_cc", openvino_cc)
    results = glob.glob(f"{out}*.npz")
    assert len(results) == 2, f"Too much or too few .npz files: {len(results)}"
    reference_results = np.load(results[0])
    inference_results = np.load(results[1])
    assert sorted(reference_results.keys()) == sorted(inference_results.keys()), \
        "Results have different number of layers"
    for layer in reference_results.keys():
        assert np.allclose(reference_results[layer], inference_results[layer], tolerance), \
            "Reference and inference results differ"
