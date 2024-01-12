// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/ops.hpp"

namespace ov {
namespace opset19 {
#define _OPENVINO_OP_REG(a, b) using b::a;
#include "openvino/opsets/opset19_tbl.hpp"
#undef _OPENVINO_OP_REG
}  // namespace opset13
}  // namespace ov
