# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import importlib
from pathlib import Path

import openvino

LIBS_ROOT = Path(__file__).resolve().parents[6] / 'thirdparty'
sys.path.append(str(LIBS_ROOT / 'open_model_zoo' / 'tools' / 'accuracy_checker'))
# pylint: disable=E0611,C0413,C0411,E0401
importlib.reload(openvino)

from openvino.tools.accuracy_checker.evaluators.quantization_model_evaluator import create_model_evaluator
from openvino.tools.accuracy_checker.config import ConfigReader
from openvino.tools.accuracy_checker.dataset import Dataset
from openvino.tools.accuracy_checker.logging import _DEFAULT_LOGGER_NAME
from openvino.tools.accuracy_checker.dataset import DataProvider as DatasetWrapper
