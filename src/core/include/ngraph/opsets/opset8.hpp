// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef IN_OV_LIBRARY
#    warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#endif

#include "ngraph/ops.hpp"

namespace ngraph {
namespace opset8 {
#define NGRAPH_OP(a, b) using b::a;
#include "ngraph/opsets/opset8_tbl.hpp"
#undef NGRAPH_OP
}  // namespace opset8
}  // namespace ngraph
