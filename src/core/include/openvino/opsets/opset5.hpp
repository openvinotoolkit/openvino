// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/ops.hpp"
#ifdef IN_OV_CORE_LIBRARY
#    error("openvino/opsets/opset5.hpp is for external use only")
#endif

namespace ov {
namespace opset5 {
#define _OPENVINO_OP_REG(a, b) using b::a;
#include "openvino/opsets/opset5_tbl.hpp"
#undef _OPENVINO_OP_REG
}  // namespace opset5
}  // namespace ov
