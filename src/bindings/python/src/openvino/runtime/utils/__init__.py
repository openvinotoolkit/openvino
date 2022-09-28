# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generic utilities. Factor related functions out to separate files."""

from openvino.pyopenvino.util import numpy_to_c
from openvino.pyopenvino.util import clone_model
from openvino.pyopenvino.util import get_constant_from_source, replace_node, replace_output_update_name
