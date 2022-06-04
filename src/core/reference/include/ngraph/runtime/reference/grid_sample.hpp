// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <cfenv>
#include <cmath>

#include "ngraph/shape.hpp"
#include "openvino/op/grid_sample.hpp"
// #include "ngraph/type/element_type.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
namespace {

using index_4D_t = std::array<size_t, 4>;

template <typename T>
T& get_v(T* buffer, const Shape& shape, const index_4D_t& index) {
    assert(shape.size() == index.size());
    size_t s = 1;
    auto offset = index.back();
    for (auto i = index.size() - 2; i; --i) {
        s *= shape[i + 1];
        offset += index[i] * s;
    }
    return *(buffer + offset);
}

template <typename GRID_ET>
GRID_ET denormalize(GRID_ET v, GRID_ET L, bool align) {
    if (align)
        return (v + 1) * (L - 1) / 2;
    else
        return ((v + 1) * L - 1) / 2;
}

template <typename GRID_ET>
std::array<GRID_ET, 2> denormalize_2D(const Shape& data_shape, GRID_ET y_n, GRID_ET x_n, bool align) {
    const auto H = static_cast<GRID_ET>(data_shape[2]);
    const auto W = static_cast<GRID_ET>(data_shape[3]);
    const auto y_d = denormalize(y_n, H, align);
    const auto x_d = denormalize(x_n, W, align);
    return {y_d, x_d};
}

template <typename DATA_ET>
using padding_function_t = std::function<DATA_ET(size_t, size_t, const DATA_ET*, const Shape&, long, long)>;

template <typename DATA_ET>
DATA_ET zeros_padding(const size_t n,
                      const size_t c,
                      const DATA_ET* data,
                      const Shape& data_shape,
                      const long y_d,
                      const long x_d) {
    const auto H = static_cast<long>(data_shape[2]);
    const auto W = static_cast<long>(data_shape[3]);
    if (y_d < 0 || x_d < 0 || y_d >= H || x_d >= W) {
        return 0;
    } else {
        const auto y = static_cast<size_t>(y_d);
        const auto x = static_cast<size_t>(x_d);
        return get_v(data, data_shape, index_4D_t{n, c, y, x});
    }
}

template <typename DATA_ET>
DATA_ET border_padding(const size_t n,
                       const size_t c,
                       const DATA_ET* data,
                       const Shape& data_shape,
                       const long y_d,
                       const long x_d) {
    const auto H = static_cast<long>(data_shape[2]);
    const auto W = static_cast<long>(data_shape[3]);
    const auto y = static_cast<size_t>(std::min(std::max(y_d, 0l), H - 1));
    const auto x = static_cast<size_t>(std::min(std::max(x_d, 0l), W - 1));
    return get_v(data, data_shape, index_4D_t{n, c, y, x});
}

template <typename DATA_ET>
DATA_ET reflection_data_no_align(const size_t n,
                                 const size_t c,
                                 const DATA_ET* data,
                                 const Shape& data_shape,
                                 long y_d,
                                 long x_d) {
    const auto H = static_cast<long>(data_shape[2]);
    const auto W = static_cast<long>(data_shape[3]);
    y_d %= 2 * H;
    x_d %= 2 * W;
    const auto y = static_cast<size_t>(y_d > H ? 2 * H - 1 - y_d : y_d);
    const auto x = static_cast<size_t>(x_d > W ? 2 * W - 1 - x_d : x_d);
    return get_v(data, data_shape, index_4D_t{n, c, y, x});
}

template <typename DATA_ET>
DATA_ET reflection_data_with_align(const size_t n,
                                   const size_t c,
                                   const DATA_ET* data,
                                   const Shape& data_shape,
                                   long y_d,
                                   long x_d) {
    const auto H = static_cast<long>(data_shape[2]);
    const auto W = static_cast<long>(data_shape[3]);
    const auto H_2_2 = 2 * (H - 1);
    const auto W_2_2 = 2 * (W - 1);
    y_d = std::abs(y_d) % H_2_2;
    x_d = std::abs(x_d) % W_2_2;
    const auto y = static_cast<size_t>(y_d > H_2_2 ? H_2_2 - y_d : y_d);
    const auto x = static_cast<size_t>(x_d > W_2_2 ? W_2_2 - x_d : x_d);
    return get_v(data, data_shape, index_4D_t{n, c, y, x});
}

template <typename DATA_ET, typename GRID_ET>
DATA_ET nearest(const size_t n,
                const size_t c,
                const DATA_ET* data,
                const Shape& data_shape,
                GRID_ET y_n,
                GRID_ET x_n,
                bool align,
                const padding_function_t<DATA_ET>& padding_func) {
    const auto p_d = denormalize_2D(data_shape, y_n, x_n, align);
    const auto y_nearest = std::lrint(p_d[0]);
    const auto x_nearest = std::lrint(p_d[1]);
    return padding_func(n, c, data, data_shape, y_nearest, x_nearest);
}

}  // namespace

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

    const auto N = data_shape[0];
    const auto C = data_shape[1];
    const auto H_out = grid_shape[1];
    const auto W_out = grid_shape[2];
    const Shape output_shape{N, C, H_out, W_out};

    const auto prev_rounding_mode = std::fegetround();
    std::fesetround(FE_TONEAREST);

    padding_function_t<DATA_ET> padding_func;
    switch (padding_mode) {
    default:
    case ov::op::v9::GridSample::PaddingMode::ZEROS:
        padding_func = zeros_padding<DATA_ET>;
        break;
    case ov::op::v9::GridSample::PaddingMode::BORDER:
        padding_func = border_padding<DATA_ET>;
        break;
    case ov::op::v9::GridSample::PaddingMode::REFLECTION:
        if (align_corners)
            padding_func = reflection_data_with_align<DATA_ET>;
        else
            padding_func = reflection_data_no_align<DATA_ET>;
        break;
    }

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t y = 0; y < H_out; ++y) {
                for (size_t x = 0; x < W_out; ++x) {
                    // const auto i = n * C + c * H_out + y * W_out + x;
                    const auto y_n = get_v(grid, grid_shape, index_4D_t{n, y, x, 1});
                    const auto x_n = get_v(grid, grid_shape, index_4D_t{n, y, x, 0});

                    const index_4D_t idx{n, c, y, x};
                    auto& out = get_v(output, output_shape, idx);

                    switch (interpolation_mode) {
                    case ov::op::v9::GridSample::InterpolationMode::BILINEAR:
                        out = 77;
                        break;
                    case ov::op::v9::GridSample::InterpolationMode::NEAREST:
                        out = nearest<DATA_ET, GRID_ET>(n, c, data, data_shape, y_n, x_n, align_corners, padding_func);
                        break;
                    case ov::op::v9::GridSample::InterpolationMode::BICUBIC:
                        out = 77;
                        break;
                    }
                }
            }
        }
    }

    std::fesetround(prev_rounding_mode);
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
