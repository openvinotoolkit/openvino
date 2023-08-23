# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

# supported_devices : CPU, GPU, GNA
test_device = os.environ.get('TEST_DEVICE', 'CPU;GPU').split(';')
