# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generic utilities. Factor related functions out to separate files."""

from openvino._pyopenvino.util import numpy_to_c
from openvino._pyopenvino.util import get_constant_from_source, replace_node, replace_output_update_name
from openvino.runtime.utils.util import clone_model
