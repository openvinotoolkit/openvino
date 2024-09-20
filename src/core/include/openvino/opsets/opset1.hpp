// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/ops.hpp"

namespace ov {
namespace opset1 {
#define _OPENVINO_OP_REG(a, b) using b::a;
OPENVINO_SUPPRESS_DEPRECATED_START
#include "openvino/opsets/opset1_tbl.hpp"
OPENVINO_SUPPRESS_DEPRECATED_END
#undef _OPENVINO_OP_REG
}  // namespace opset1
}  // namespace ov
