# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import logging

from path_utils import get_lib_path  # pylint: disable=import-error


def get_lib_sizes(path, libraries):
    """Function for getting lib sizes by lib names"""
    assert Path.exists(path), f'Directory {path} isn\'t created'
    result = {}
    error_lib = []
    for lib in libraries:
        try:
            result[lib] = Path(path).joinpath(get_lib_path(lib)).stat().st_size
        except FileNotFoundError as error:
            error_lib.append(str(error))
    assert len(error_lib) == 0, 'Following libraries couldn\'t be found: \n{}'.format('\n'.join(error_lib))
    return result


def test_size_tracking_libs(openvino_root_dir, test_id, model, artifacts):
    log = logging.getLogger('size_tracking')
    libraries = ['inference_engine_transformations', 'MKLDNNPlugin', 'ngraph']

    ref_libs_size = get_lib_sizes(openvino_root_dir, libraries)
    install_prefix = artifacts / test_id / 'install_pkg'
    lib_sizes = get_lib_sizes(install_prefix, libraries)

    for lib in libraries:
        lib_size_diff = ref_libs_size[lib] - lib_sizes[lib]
        lib_size_diff_percent = lib_size_diff / ref_libs_size[lib] * 100
        log.info('{}: old - {}kB; new - {}kB; diff = {}kB({:.2f}%)'.format(lib,
                                                                           ref_libs_size[lib] / 1024,
                                                                           lib_sizes[lib] / 1024,
                                                                           lib_size_diff / 1024,
                                                                           lib_size_diff_percent))
    res = [lib for lib in libraries if lib_sizes[lib] > ref_libs_size[lib]]
    assert len(res) == 0, f'These libraries: {res} have increased in size!'
