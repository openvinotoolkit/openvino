# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generic utilities. Factor related functions out to separate files."""

from openvino.pyopenvino import util

numpy_to_c = util.numpy_to_c
get_constant_from_source = util.get_constant_from_source
