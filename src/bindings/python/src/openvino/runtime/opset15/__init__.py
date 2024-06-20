# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Inlcudes new operators added in Opset15

# TODO (ticket 138273): Add previous opset operators at the end of opset15 development
from openvino.runtime.opset1.ops import parameter
from openvino.runtime.opset15.ops import col2im
from openvino.runtime.opset15.ops import embedding_bag_offsets
from openvino.runtime.opset15.ops import embedding_bag_packed
from openvino.runtime.opset15.ops import scatter_nd_update
