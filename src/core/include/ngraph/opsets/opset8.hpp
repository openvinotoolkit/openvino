// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ops.hpp"

namespace ngraph {
namespace opset8 {
#define NGRAPH_OP(a, b) using b::a;
#include "ngraph/opsets/opset8_tbl.hpp"
#undef NGRAPH_OP
}  // namespace opset8
}  // namespace ngraph
