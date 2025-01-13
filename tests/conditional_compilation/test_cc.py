#!/usr/bin/env python3

# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Test conditional compilation.
"""

import glob
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from proc_utils import cmd_exec  # pylint: disable=import-error

from test_utils import get_lib_sizes, infer_tool, make_build, run_infer  # pylint: disable=import-error

log = logging.getLogger()


@pytest.mark.dependency(name="cc_collect")
def test_cc_collect(test_id, prepared_models, openvino_ref, test_info,
                    save_session_info, sea_runtool, collector_dir, artifacts):  # pylint: disable=unused-argument
    """Test conditional compilation statistics collection
    :param test_info: custom `test_info` field of built-in `request` pytest fixture.
                      contain a dictionary to store test metadata.
    """
    out = artifacts / test_id
    infer_out_dir = out / "inference_result"
    test_info["test_id"] = test_id

    # cleanup old data if any
    prev_result = glob.glob(f"{out / test_id}.pid*.csv")
    for path in prev_result:
        os.remove(path)
    # Create a directory for  infer results, if it haven't done before
    infer_out_dir.mkdir(parents=True, exist_ok=True)
    # run use case
    return_code, output = cmd_exec(
        [
            sys.executable,
            str(sea_runtool),
            f"--output={out / test_id}",
            f"--bindir={collector_dir}",
            "!",
            sys.executable,
            infer_tool,
            *[f"-m={model}" for model in prepared_models],
            "-d=CPU",
            f"-r={infer_out_dir}"
        ]
    )
    out_csv = glob.glob(f"{out / test_id}.pid*.csv")
    test_info["out_csv"] = out_csv

    assert return_code == 0, f"Command exited with non-zero status {return_code}:\n {output}"
    assert len(out_csv) == 1, f'Multiple or none "{out / test_id}.pid*.csv" files'


@pytest.mark.dependency(depends=["cc_collect"])
def test_minimized_pkg(test_id, models, openvino_root_dir, artifacts):  # pylint: disable=unused-argument
    """Build and install OpenVINO package with collected conditional compilation statistics."""
    out = artifacts / test_id
    install_prefix = out / "install_pkg"
    build_dir = openvino_root_dir / "build_minimized"

    out_csv = glob.glob(f"{out / test_id}.pid*.csv")
    assert len(out_csv) == 1, f'Multiple or none "{out / test_id}.pid*.csv" files'

    log.info("Building minimized build at %s", build_dir)

    return_code, output = make_build(
        openvino_root_dir,
        build_dir,
        install_prefix,
        cmake_additional_args=[f"-DSELECTIVE_BUILD_STAT={out_csv[0]}"],
        log=log,
    )
    assert return_code == 0, f"Command exited with non-zero status {return_code}:\n {output}"


@pytest.mark.dependency(depends=["cc_collect", "minimized_pkg"])
def test_infer(test_id, prepared_models, artifacts):
    """Test inference with conditional compiled binaries."""
    out = artifacts / test_id
    minimized_pkg = out / "install_pkg"
    infer_out_dir_cc = out / "inference_result_cc/"

    return_code, output = run_infer(prepared_models, infer_out_dir_cc, minimized_pkg)
    assert return_code == 0, f"Command exited with non-zero status {return_code}:\n {output}"


@pytest.mark.dependency(depends=["cc_collect", "minimized_pkg"])
def test_verify(test_id, prepared_models, openvino_ref, artifacts, tolerance=1e-6):  # pylint: disable=too-many-arguments
    """Test verifying that inference results are equal."""
    out = artifacts / test_id
    minimized_pkg = out / "install_pkg"

    infer_out_dir_cc = out / "inference_result_cc/"
    infer_out_dir = out / "inference_result/"

    return_code, output = run_infer(prepared_models, infer_out_dir, openvino_ref)
    assert return_code == 0, f"Command exited with non-zero status {return_code}:\n {output}"
    return_code, output = run_infer(prepared_models, infer_out_dir_cc, minimized_pkg)
    assert return_code == 0, f"Command exited with non-zero status {return_code}:\n {output}"

    for model in prepared_models:
        out_file = f"{infer_out_dir / Path(model).name}.npz"
        out_file_cc = f"{infer_out_dir_cc / Path(model).name}.npz"

        reference_results = dict(np.load(out_file))
        inference_results = dict(np.load(out_file_cc))
        assert sorted(reference_results.keys()) == sorted(
            inference_results.keys()
        ), "Results have different number of layers"
        for layer in reference_results.keys():
            assert np.allclose(
                reference_results[layer], inference_results[layer], tolerance
            ), "Reference and inference results differ"


@pytest.mark.dependency(depends=["cc_collect", "minimized_pkg"])
def test_libs_size(test_id, models, openvino_ref, artifacts):  # pylint: disable=unused-argument
    """Test if libraries haven't increased in size after conditional compilation."""
    libraries = ["openvino", "openvino_intel_cpu_plugin"]
    minimized_pkg = artifacts / test_id / "install_pkg"
    ref_libs_size = get_lib_sizes(openvino_ref, libraries)
    lib_sizes = get_lib_sizes(minimized_pkg, libraries)

    for lib in libraries:
        lib_size_diff = ref_libs_size[lib] - lib_sizes[lib]
        lib_size_diff_percent = lib_size_diff / ref_libs_size[lib] * 100
        log.info(
            "{}: old - {}kB; new - {}kB; diff = {}kB({:.2f}%)".format(
                lib,
                ref_libs_size[lib] / 1024,
                lib_sizes[lib] / 1024,
                lib_size_diff / 1024,
                lib_size_diff_percent,
            )
        )
    res = [lib for lib in libraries if lib_sizes[lib] > ref_libs_size[lib]]
    assert len(res) == 0, f"These libraries: {res} have increased in size!"
