// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef IN_OPENVINO_LIBRARY
#    error("openvino/opsets/opset4.hpp is for external use only")
#endif

#include "openvino/op/ops.hpp"

namespace ov {
namespace opset4 {
#define _OPENVINO_OP_REG(a, b) using b::a;
#include "openvino/opsets/opset4_tbl.hpp"
#undef _OPENVINO_OP_REG
}  // namespace opset4
}  // namespace ov
