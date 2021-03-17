#!/usr/bin/env python3
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Test to verify inference results.
"""

import numpy as np
import os
import glob


def test_verify(test_id, model, artifacts):
    out = os.path.join(*list(artifacts.parts), model.stem)
    # cleanup old data if any
    results = glob.glob(f"{out}*.npz")
    assert len(results) == 2, "Too much or too few .npz files"
    reference_results = np.load(results[0])
    inference_results = np.load(results[1])
    assert reference_results.keys() == inference_results.keys(), "Results have different layers"
    for layer in reference_results.keys():
        # print(np.absolute(reference_results[layer] - inference_results[layer]))
        assert np.allclose(reference_results[layer], inference_results[layer], 0.000001) is True, \
            "Reference and inference results differ"

