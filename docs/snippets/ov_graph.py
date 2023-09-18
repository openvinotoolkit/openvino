# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


#! [ov:graph]
# _____________    _____________
# | Parameter |    | Parameter |
# |   data1   |    |   data2   |
# |___________|    |___________|
#         |            |
# data1_t |            | data2_t
#          \          /
#           \        /
#            \      /
#         ____\____/____
#         |   Concat   |
#         |   concat   |
#         |____________|
#               |
#               | concat_t
#               |
#        _______|_______
#        |    Result   |
#        |    result   |
#        |_____________|

import openvino as ov
import openvino.runtime.opset12 as ops


data1 = ops.parameter([1, 3, 2, 2], ov.Type.i64)
data1.friendly_name = "data1"      # operation name
data1.output(0).name = "data1_t" # tensor name
data2 = ops.parameter([1, 2, 2, 2], ov.Type.i64)
data2.friendly_name = "data2"      # operation name
data2.output(0).name = "data2_t"   # tensor name

concat = ops.concat([data1, data2], 1)
concat.friendly_name = "concat"    # operation name
concat.output(0).name = "concat_t" # tensor name

result = ops.result(concat)
result.friendly_name = "result"    # operation name

model = ov.Model(result, [data1, data2], "model_name")
#! [ov:graph]
