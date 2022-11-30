// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#ifdef IN_OPENVINO_LIBRARY
#    error("ngraph/opsets/opset5.hpp is for external use only")
#endif

#include "ngraph/ops.hpp"

namespace ngraph {
namespace opset5 {
#define NGRAPH_OP(a, b) using b::a;
#include "ngraph/opsets/opset5_tbl.hpp"
#undef NGRAPH_OP
}  // namespace opset5
}  // namespace ngraph
