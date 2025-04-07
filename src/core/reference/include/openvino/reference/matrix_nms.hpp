// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/op/matrix_nms.hpp"

namespace ov {
namespace reference {
void matrix_nms(const float* boxes_data,
                const Shape& boxes_data_shape,
                const float* scores_data,
                const Shape& scores_data_shape,
                const op::v8::MatrixNms::Attributes& attrs,
                float* selected_outputs,
                const Shape& selected_outputs_shape,
                int64_t* selected_indices,
                const Shape& selected_indices_shape,
                int64_t* valid_outputs);

}  // namespace reference
}  // namespace ov
