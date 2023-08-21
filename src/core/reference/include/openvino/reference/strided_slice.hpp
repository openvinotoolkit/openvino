// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "ngraph/check.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/opt_kernel/reshape.hpp"
#include "ngraph/slice_plan.hpp"
#include "openvino/reference/reverse.hpp"
#include "openvino/reference/slice.hpp"

namespace ov {
namespace reference {
NGRAPH_SUPPRESS_DEPRECATED_START
void strided_slice(const char* arg, char* out, const Shape& arg_shape, const ngraph::SlicePlan& sp, size_t elem_type);
NGRAPH_SUPPRESS_DEPRECATED_END
}  // namespace reference
}  // namespace ov
