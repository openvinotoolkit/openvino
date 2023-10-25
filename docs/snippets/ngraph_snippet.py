# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


#! [ngraph:graph]
# _____________    _____________
# | Parameter |    | Parameter |
# |   data1   |    |   data2   |
# |___________|    |___________|
#         |            |
#         |            |
#          \          /
#           \        /
#            \      /
#         ____\____/____
#         |   Concat   |
#         |   concat   |
#         |____________|
#               |
#               |
#               |
#        _______|_______
#        |    Result   |
#        |    result   |
#        |_____________|

import ngraph as ng
import numpy as np


data1 = ng.opset8.parameter([1, 3, 2, 2], np.int64)
data1.friendly_name = "data1" # operation name
data2 = ng.opset8.parameter([1, 2, 2, 2], np.int64)
data2.friendly_name = "data2" # operation name

concat = ng.opset8.concat([data1, data2], 1)
concat.friendly_name = "concat" # operation name

result = ng.opset8.result(concat)
result.friendly_name = "result" # operation name

f = ng.Function(result, [data1, data2], "function_name")
#! [ngraph:graph]
