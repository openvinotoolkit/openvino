// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/shape.hpp"
#include "openvino/op/grid_sample.hpp"
// #include "ngraph/type/element_type.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
template <typename T, typename U>
void grid_sample(T* output,
                 const T* data,
                 const U* grid,
                 const Shape& data_shape,
                 const Shape& grid_shape,
                 const bool align_corners,
                 const ov::op::v9::GridSample::InterpolationMode interpolation_mode,
                 const ov::op::v9::GridSample::PaddingMode padding_mode) {
    const T v = align_corners ? 7 : 4;
    const auto o_end = output + data_shape[0] * data_shape[1] * grid_shape[2] * grid_shape[3];
    // auto p = output;
    for (; output < o_end; ++output)
        *output = v;
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
