# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import mo


TEST_ROOT = Path(__file__).parent.parent.absolute()
PROJECT_ROOT = TEST_ROOT.parent
LIBS_ROOT = PROJECT_ROOT / 'thirdparty'
MO_PATH = Path(mo.__file__).parent.parent
AC_PATH = LIBS_ROOT / 'open_model_zoo' / 'tools' / 'accuracy_checker'

MODELS_PATH = TEST_ROOT / 'data' / 'models'
REFERENCE_MODELS_PATH = TEST_ROOT / 'data' / 'reference_models'
HARDWARE_CONFIG_PATH = TEST_ROOT / 'data' / 'hardware_configs'
HARDWARE_CONFIG_REFERENCE_PATH = HARDWARE_CONFIG_PATH / 'reference'

TOOL_CONFIG_PATH = TEST_ROOT / 'data' / 'tool_configs'
ENGINE_CONFIG_PATH = TEST_ROOT / 'data' / 'engine_configs'
DATASET_CONFIG_PATH = TEST_ROOT / 'data' / 'datasets'
TELEMETRY_CONFIG_PATH = TEST_ROOT / 'data' / 'telemetry'

INTERMEDIATE_CONFIG_PATH = TEST_ROOT / 'data' / 'intermediate_configs'
