// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/shape.hpp"
#include "openvino/op/grid_sample.hpp"
// #include "ngraph/type/element_type.hpp"

namespace {

template <typename DATA_ET, typename GRID_ET>
DATA_ET nearest() {
    return 37;
}

}  // namespace

namespace ngraph {
namespace runtime {
namespace reference {
template <typename DATA_ET, typename GRID_ET>
void grid_sample(DATA_ET* output,
                 const DATA_ET* data,
                 const GRID_ET* grid,
                 const Shape& data_shape,
                 const Shape& grid_shape,
                 const bool align_corners,
                 const ov::op::v9::GridSample::InterpolationMode interpolation_mode,
                 const ov::op::v9::GridSample::PaddingMode padding_mode) {
    // assert(len(data.shape) == 4 and len(grid.shape) == 4)
    // assert(data.shape[0] == grid.shape[0] and grid.shape[3] == 2)

    auto N = data_shape[0];
    auto C = data_shape[1];
    auto H_out = grid_shape[0];
    auto W_out = grid_shape[0];

    for (decltype(N) n = 0; n < N; ++n)
        for (decltype(C) c = 0; c < C; ++c)
            for (decltype(H_out) h = 0; h < H_out; ++h)
                for (decltype(W_out) w = 0; w < W_out; ++w) {
                    const size_t i = n * C + c * H_out + h * W_out + w;
                    switch (interpolation_mode) {
                    case ov::op::v9::GridSample::InterpolationMode::BILINEAR:
                        output[i] = 77;
                        break;
                    case ov::op::v9::GridSample::InterpolationMode::NEAREST:
                        output[i] = nearest<DATA_ET, GRID_ET>();
                        break;
                    case ov::op::v9::GridSample::InterpolationMode::BICUBIC:
                        output[i] = 77;
                        break;
                    }
                }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
