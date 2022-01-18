# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from openvino.tools.mo import mo


TEST_ROOT = Path(__file__).parent.parent.absolute()
LIBS_ROOT = Path(__file__).resolve().parents[4] / 'thirdparty'
MO_PATH = Path(mo.__file__).parent

MODELS_PATH = TEST_ROOT / 'data' / 'models'
REFERENCE_MODELS_PATH = TEST_ROOT / 'data' / 'reference_models'
HARDWARE_CONFIG_PATH = TEST_ROOT / 'data' / 'hardware_configs'
HARDWARE_CONFIG_REFERENCE_PATH = HARDWARE_CONFIG_PATH / 'reference'

TOOL_CONFIG_PATH = TEST_ROOT / 'data' / 'tool_configs'
ENGINE_CONFIG_PATH = TEST_ROOT / 'data' / 'engine_configs'
DATASET_CONFIG_PATH = TEST_ROOT / 'data' / 'datasets'
TELEMETRY_CONFIG_PATH = TEST_ROOT / 'data' / 'telemetry'

INTERMEDIATE_CONFIG_PATH = TEST_ROOT / 'data' / 'intermediate_configs'
