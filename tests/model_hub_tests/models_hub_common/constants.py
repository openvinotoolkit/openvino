# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

'''
@brief Time in seconds of measurement performance on each of the networks. This time doesn't include
loading and heating and includes measurement only one of 2 models - got through convert and read_model.
Both "converted" and "read_model" modes will be 2 * runtime_measure_duration 
'''
precommit_runtime_measure_duration = os.environ.get('PRECOMMIT_RUNTIME_MEASURE_DURATION', '60')
nightly_runtime_measure_duration = os.environ.get('NIGHTLY_RUNTIME_MEASURE_DURATION', '15')
'''
@brief Time in seconds of heating before measurement
'''
precommit_runtime_heat_duration = os.environ.get('PRECOMMIT_RUNTIME_HEAT_DURATION', '5')
nigtly_runtime_heat_duration = os.environ.get('NIGHTLY_RUNTIME_HEAT_DURATION', '5')

tf_hub_cache_dir = os.environ.get('TFHUB_CACHE_DIR',
                                  os.path.join(tempfile.gettempdir(), "tfhub_modules"))
hf_cache_dir = os.environ.get('HF_HUB_CACHE',
                              os.path.join(tempfile.gettempdir(), "hugging_face"))
# update environment variables if temporal directory is used
os.environ['TFHUB_CACHE_DIR'] = tf_hub_cache_dir
os.environ['HF_HUB_CACHE'] = hf_cache_dir

no_clean_cache_dir = False
hf_hub_cache_dir = tempfile.gettempdir()
if os.environ.get('USE_SYSTEM_CACHE', 'True') == 'False':
    no_clean_cache_dir = True
    os.environ['HUGGINGFACE_HUB_CACHE'] = hf_hub_cache_dir

# supported_devices : CPU, GPU
test_device = os.environ.get('TEST_DEVICE', 'CPU;GPU').split(';')
