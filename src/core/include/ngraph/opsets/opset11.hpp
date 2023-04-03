// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ops.hpp"

namespace ngraph {
namespace opset11 {
#define NGRAPH_OP(a, b) using b::a;
#include "ngraph/opsets/opset11_tbl.hpp"
#undef NGRAPH_OP
}  // namespace opset11
}  // namespace ngraph
