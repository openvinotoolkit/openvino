# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

__path__ = __import__('pkgutil').extend_path(__path__, __name__)

# version
from openvino.pyopenvino import get_version

__version__ = get_version() 
