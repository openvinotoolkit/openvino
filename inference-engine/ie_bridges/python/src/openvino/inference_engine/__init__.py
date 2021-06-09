# -*- coding: utf-8 -*-
# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .ie_api import *

__all__ = ['IENetwork', 'TensorDesc', 'IECore', 'Blob', 'PreProcessInfo', 'get_version']
__version__ = get_version()  # type: ignore
