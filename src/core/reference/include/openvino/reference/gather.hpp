// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <numeric>

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
void gather(const char* const data,
            const int64_t* const indices,
            char* out,
            const Shape& data_shape,
            const Shape& indices_shape,
            const Shape& out_shape,
            size_t axis,
            size_t element_size,
            size_t batch_dims = 0);
}  // namespace reference
}  // namespace ov
