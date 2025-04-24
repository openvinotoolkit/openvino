// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/transpose.hpp"

#include <cfenv>
#include <cmath>
#include <numeric>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/reference/reshape.hpp"

namespace ov {
namespace reference {
void transpose(const char* data,
               char* out,
               const Shape& data_shape,
               size_t element_size,
               const std::vector<int64_t>& axes_order,
               const Shape& out_shape) {
    // To reuse reference::reshape axes order vector has to be converted to AxisVector
    // Negative axes are not supported, it is validated by transpose evaluate method
    const AxisVector axes_vector(axes_order.begin(), axes_order.end());
    reshape(data, out, data_shape, axes_vector, out_shape, element_size);
}
}  // namespace reference
}  // namespace ov
