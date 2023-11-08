# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

wget_cache_dir = os.environ.get('WGET_CACHE_DIR', os.path.join(tempfile.gettempdir(), "wget_cache"))

tf_hub_cache_dir = os.environ.get('TFHUB_CACHE_DIR',
                                  os.path.join(tempfile.gettempdir(), "tfhub_modules"))
os.environ['TFHUB_CACHE_DIR'] = tf_hub_cache_dir

hf_hub_cache_dir = tempfile.gettempdir()
if os.environ.get('USE_SYSTEM_CACHE', 'True') == 'False':
    os.environ['HUGGINGFACE_HUB_CACHE'] = hf_hub_cache_dir

# supported_devices : CPU, GPU, GNA
test_device = os.environ.get('TEST_DEVICE', 'CPU;GPU').split(';')
