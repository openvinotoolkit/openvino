// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "openvino/reference/slice.hpp"

namespace ngraph {
namespace reference {
void split(const char* data,
           const Shape& data_shape,
           size_t elem_size,
           int64_t axis,
           size_t num_splits,
           char** out_data);
}
}  // namespace ngraph
