# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import unittest

import numpy as np
from generator import generator, generate

import openvino.runtime.opset11 as opset11
from openvino.runtime import Model
from openvino.runtime import PartialShape, Dimension
from openvino.tools.ovc.convert import convert_model
from openvino.tools.ovc.error import Error
