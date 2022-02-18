// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ops.hpp"

namespace ngraph {
namespace opset7 {
#define NGRAPH_OP(a, b) using b::a;
#include "ngraph/opsets/opset7_tbl.hpp"
#undef NGRAPH_OP
}  // namespace opset7
}  // namespace ngraph
