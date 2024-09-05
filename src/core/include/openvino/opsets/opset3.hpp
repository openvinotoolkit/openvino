// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/ops.hpp"

namespace ov {
namespace opset3 {
#define _OPENVINO_OP_REG(a, b) using b::a;
OPENVINO_SUPPRESS_DEPRECATED_START
#include "openvino/opsets/opset3_tbl.hpp"
OPENVINO_SUPPRESS_DEPRECATED_END
#undef _OPENVINO_OP_REG
}  // namespace opset3
}  // namespace ov
