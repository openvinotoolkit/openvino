// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ops.hpp"

namespace ngraph {
namespace opset1 {
#define NGRAPH_OP(a, b) using b::a;
#include "ngraph/opsets/opset1_tbl.hpp"
#undef NGRAPH_OP
}  // namespace opset1
}  // namespace ngraph
