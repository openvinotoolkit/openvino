# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.pot.benchmark.benchmark import benchmark_embedded, set_benchmark_config
from openvino.tools.pot.utils.logger import get_logger, init_logger
from .utils.path import TEST_ROOT


logger = get_logger(__name__)
REFERENCE_MODELS_PATH = TEST_ROOT/'../thirdparty/open_model_zoo/tools/accuracy_checker/data/test_models/SampLeNet.xml'

def test_benchmark(model=None, cfg=None):
    init_logger(level='INFO')
    if cfg:
        set_benchmark_config(cfg)
    if model:
        benchmark_embedded(model=model)
        return

    path_to_model_file = str(REFERENCE_MODELS_PATH)
    logger.info('Benchmark test with {}'.format(path_to_model_file))

    cfg = {'nireq': 0}
    set_benchmark_config(cfg)
    benchmark_embedded(model=None, mf=path_to_model_file, duration_seconds=1)

    cfg = {'nireq': 0, 'benchmark_app_dir':""}
    set_benchmark_config(cfg)
    benchmark_embedded(model=None, mf=path_to_model_file, duration_seconds=1)

    cfg = {'nireq': 0, 'benchmark_app_dir':"wrong_benchmark_dir"}
    set_benchmark_config(cfg)
    benchmark_embedded(model=None, mf=path_to_model_file, duration_seconds=1)
