#!/usr/bin/env python3
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Test to verify inference results.
"""

import numpy as np
import glob


def test_verify(test_id, model, artifacts, inaccuracy=0.000001):
    out = artifacts / test_id
    results = glob.glob(f"{out}*.npz")
    assert len(results) == 2, "Too much or too few .npz files"
    reference_results = np.load(results[0])
    inference_results = np.load(results[1])
    assert reference_results.keys() == inference_results.keys(), "Results have different layers"
    for layer in reference_results.keys():
        assert np.allclose(reference_results[layer], inference_results[layer], inaccuracy) is True, \
            "Reference and inference results differ"
