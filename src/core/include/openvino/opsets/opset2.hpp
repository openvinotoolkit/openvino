// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef IN_OPENVINO_LIBRARY
#    error("openvino/opsets/opset2.hpp is for external use only")
#endif

#include "openvino/op/ops.hpp"

namespace ov {
namespace opset2 {
#define _OPENVINO_OP_REG(a, b) using b::a;
#include "openvino/opsets/opset2_tbl.hpp"
#undef _OPENVINO_OP_REG
}  // namespace opset2
}  // namespace ov
