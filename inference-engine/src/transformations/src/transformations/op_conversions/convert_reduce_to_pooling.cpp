// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertReduceToPooling, "ConvertReduceToPooling", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertReduceMeanToPooling, "ConvertReduceMeanToPooling", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertReduceMaxToPooling, "ConvertReduceMaxToPooling", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertReduceSumToPooling, "ConvertReduceSumToPooling", 0);
