# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

if os.environ.get('TFHUB_CACHE_DIR') is not None:
    tf_hub_cache_dir = os.environ['TFHUB_CACHE_DIR']
else:
    tf_hub_cache_dir = os.path.join(tempfile.gettempdir(), "tfhub_modules")
    os.environ['TFHUB_CACHE_DIR'] = tf_hub_cache_dir

# supported_devices : CPU, GPU, GNA
test_device = os.environ.get('TEST_DEVICE', 'CPU;GPU').split(';')
