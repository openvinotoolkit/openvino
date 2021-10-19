# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

try:
    from thirdparty.open_model_zoo.tools.accuracy_checker.\
        openvino.tools.accuracy_checker.evaluators.quantization_model_evaluator import create_model_evaluator
    from thirdparty.open_model_zoo.tools.accuracy_checker.openvino.tools.accuracy_checker.config import ConfigReader
    from thirdparty.open_model_zoo.tools.accuracy_checker.openvino.tools.accuracy_checker.dataset import\
        Dataset, DataProvider as DatasetWrapper
    from thirdparty.open_model_zoo.tools.accuracy_checker.openvino.tools.accuracy_checker.logging\
        import _DEFAULT_LOGGER_NAME

except ImportError:
    try:
        from openvino.tools.accuracy_checker.evaluators.quantization_model_evaluator import create_model_evaluator
        from openvino.tools.accuracy_checker.config import ConfigReader
        from openvino.tools.accuracy_checker.dataset import Dataset
        from openvino.tools.accuracy_checker.logging import _DEFAULT_LOGGER_NAME
        from openvino.tools.accuracy_checker.dataset import DataProvider as DatasetWrapper
    except ImportError:
        from accuracy_checker.evaluators.quantization_model_evaluator import create_model_evaluator
        from accuracy_checker.config import ConfigReader
        from accuracy_checker.dataset import Dataset
        from accuracy_checker.logging import _DEFAULT_LOGGER_NAME
        try:
            from accuracy_checker.dataset import DataProvider as DatasetWrapper
        except ImportError:
            from accuracy_checker.dataset import DatasetWrapper
