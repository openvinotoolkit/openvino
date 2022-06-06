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
    // In this context below assertion is guaranteed by grid_sample(..) function.
    // assert(shape.size() == index.size());
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
    const auto H_2 = static_cast<long>(data_shape[2]) * 2l;
    const auto W_2 = static_cast<long>(data_shape[3]) * 2l;
    y_d = (y_d % H_2 + H_2) % H_2;
    x_d = (x_d % W_2 + W_2) % W_2;
    const auto y = static_cast<size_t>(y_d >= H ? H_2 - 1 - y_d : y_d);
    const auto x = static_cast<size_t>(x_d >= W ? W_2 - 1 - x_d : x_d);
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
    const auto y = static_cast<size_t>(y_d >= H ? H_2_2 - y_d : y_d);
    const auto x = static_cast<size_t>(x_d >= W ? W_2_2 - x_d : x_d);
    return get_v(data, data_shape, index_4D_t{n, c, y, x});
}

// TODO use more robust stuff .. it's for POC
template <typename T>
struct square {
    T v00, v01, v10, v11;
};

// TODO use more robust stuff .. it's for POC
template <typename DATA_ET>
square<DATA_ET> get_square(const size_t n,
                           const size_t c,
                           const DATA_ET* data,
                           const Shape& data_shape,
                           long y_d,
                           long x_d,
                           const padding_function_t<DATA_ET>& padding_func) {
    square<DATA_ET> s;
    s.v00 = padding_func(n, c, data, data_shape, y_d, x_d);
    s.v01 = padding_func(n, c, data, data_shape, y_d, x_d + 1);
    s.v10 = padding_func(n, c, data, data_shape, y_d + 1, x_d);
    s.v11 = padding_func(n, c, data, data_shape, y_d + 1, x_d + 1);
    return s;
}

// TODO rename me .. sth like bili..._2D
template <typename DATA_ET>
DATA_ET calc_bilinear(const square<DATA_ET>& s, DATA_ET dy, DATA_ET dx) {
    const auto q0 = (1 - dx) * s.v00 + dx * s.v01;
    const auto q1 = (1 - dx) * s.v10 + dx * s.v11;
    return dy * q1 + (1 - dy) * q0;
}

template <typename DATA_ET, typename GRID_ET>
DATA_ET bilinear(const size_t n,
                 const size_t c,
                 const DATA_ET* data,
                 const Shape& data_shape,
                 GRID_ET y_n,
                 GRID_ET x_n,
                 bool align,
                 const padding_function_t<DATA_ET>& padding_func) {
    const auto vec_yx = denormalize_2D(data_shape, y_n, x_n, align);
    const auto y_topleft = std::floor(vec_yx[0]);
    const auto x_topleft = std::floor(vec_yx[1]);
    const auto dy = vec_yx[0] - y_topleft;
    const auto dx = vec_yx[1] - x_topleft;
    const auto s = get_square(n, c, data, data_shape, y_topleft, x_topleft, padding_func);
    return calc_bilinear(s, dy, dx);
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
    const auto vec_yx = denormalize_2D(data_shape, y_n, x_n, align);
    const auto y_nearest = std::lrint(vec_yx[0]);
    const auto x_nearest = std::lrint(vec_yx[1]);
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
    assert(data_shape.size() == 4 && grid_shape.size() == 4);
    assert(data_shape[0] == grid_shape[0] and grid_shape[3] == 2);

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
                    const auto y_n = get_v(grid, grid_shape, index_4D_t{n, y, x, 1});
                    const auto x_n = get_v(grid, grid_shape, index_4D_t{n, y, x, 0});

                    auto& out = get_v(output, output_shape, index_4D_t{n, c, y, x});

                    switch (interpolation_mode) {
                    case ov::op::v9::GridSample::InterpolationMode::BILINEAR:
                        out = bilinear<DATA_ET, GRID_ET>(n, c, data, data_shape, y_n, x_n, align_corners, padding_func);
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
