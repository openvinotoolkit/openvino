# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Properties
import openvino._pyopenvino.properties.streams as __streams

# Classes
from openvino._pyopenvino.properties.streams import Num
from openvino.properties._properties import __make_properties

__make_properties(__streams, __name__)
