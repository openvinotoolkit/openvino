// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "openvino/op/util/slice_plan.hpp"
#include "openvino/reference/reverse.hpp"
#include "openvino/reference/slice.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {
void strided_slice(const char* arg, char* out, const Shape& arg_shape, const op::util::SlicePlan& sp, size_t elem_type);
}  // namespace reference
}  // namespace ov
