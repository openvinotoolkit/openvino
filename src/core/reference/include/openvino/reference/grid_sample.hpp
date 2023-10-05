// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cfenv>
#include <cmath>

#include "openvino/core/shape.hpp"
#include "openvino/op/grid_sample.hpp"

namespace ov {
namespace reference {
namespace {

using index_4D_t = typename std::array<size_t, 4>;

template <typename GRID_ET>
using denormalize_fn_t = typename std::function<GRID_ET(GRID_ET, size_t)>;

template <typename DATA_ET>
using get_padded_fn_t = typename std::function<DATA_ET(const DATA_ET*, const Shape&, size_t, size_t, long, long)>;

template <typename T>
T& get_single_value(T* buffer, const Shape& shape, const index_4D_t& index) {
    // In this context below assertion is guaranteed by grid_sample(..) function.
    // assert(shape.size() == index.size());
    auto sx = shape.back();
    auto offset = index.back();
    for (auto i = index.size() - 2; i > 0; --i) {
        offset += index[i] * sx;
        sx *= shape[i];
    }
    offset += index[0] * sx;
    return buffer[offset];
}

template <typename GRID_ET>
GRID_ET rescale_align(GRID_ET value, size_t range) {
    return (value + 1) * (static_cast<GRID_ET>(range) - 1) / 2;
}

template <typename GRID_ET>
GRID_ET rescale_noalign(GRID_ET value, size_t range) {
    return ((value + 1) * static_cast<GRID_ET>(range) - 1) / 2;
}

template <typename DATA_ET>
DATA_ET zeros_padding(const DATA_ET* data,
                      const Shape& data_shape,
                      const size_t n,
                      const size_t c,
                      const long y_d,
                      const long x_d) {
    const auto H = static_cast<long>(data_shape[2]);
    const auto W = static_cast<long>(data_shape[3]);
    if (y_d < 0 || x_d < 0 || y_d >= H || x_d >= W) {
        return 0;
    } else {
        const auto y = static_cast<size_t>(y_d);
        const auto x = static_cast<size_t>(x_d);
        return get_single_value(data, data_shape, index_4D_t{n, c, y, x});
    }
}

template <typename DATA_ET>
DATA_ET border_padding(const DATA_ET* data,
                       const Shape& data_shape,
                       const size_t n,
                       const size_t c,
                       const long y_d,
                       const long x_d) {
    const auto H = static_cast<long>(data_shape[2]);
    const auto W = static_cast<long>(data_shape[3]);
    const auto y = static_cast<size_t>(std::min(std::max(y_d, 0l), H - 1));
    const auto x = static_cast<size_t>(std::min(std::max(x_d, 0l), W - 1));
    return get_single_value(data, data_shape, index_4D_t{n, c, y, x});
}

template <typename DATA_ET>
DATA_ET reflection_data_no_align(const DATA_ET* data,
                                 const Shape& data_shape,
                                 const size_t n,
                                 const size_t c,
                                 long y_d,
                                 long x_d) {
    const auto H = static_cast<long>(data_shape[2]);
    const auto W = static_cast<long>(data_shape[3]);
    const auto H_2 = H * 2l;
    const auto W_2 = W * 2l;
    y_d = (y_d % H_2 + H_2) % H_2;
    x_d = (x_d % W_2 + W_2) % W_2;
    const auto y = static_cast<size_t>(y_d >= H ? H_2 - 1 - y_d : y_d);
    const auto x = static_cast<size_t>(x_d >= W ? W_2 - 1 - x_d : x_d);
    return get_single_value(data, data_shape, index_4D_t{n, c, y, x});
}

template <typename DATA_ET>
DATA_ET reflection_data_with_align(const DATA_ET* data,
                                   const Shape& data_shape,
                                   const size_t n,
                                   const size_t c,
                                   long y_d,
                                   long x_d) {
    const auto H = static_cast<long>(data_shape[2]);
    const auto W = static_cast<long>(data_shape[3]);
    const auto H_2_2 = H == 1 ? 1 : 2 * (H - 1);
    const auto W_2_2 = W == 1 ? 1 : 2 * (W - 1);
    y_d = std::abs(y_d) % H_2_2;
    x_d = std::abs(x_d) % W_2_2;
    const auto y = static_cast<size_t>(y_d >= H ? H_2_2 - y_d : y_d);
    const auto x = static_cast<size_t>(x_d >= W ? W_2_2 - x_d : x_d);
    return get_single_value(data, data_shape, index_4D_t{n, c, y, x});
}

template <typename DATA_ET, typename GRID_ET>
DATA_ET bilinear(const DATA_ET* data,
                 const Shape& data_shape,
                 const size_t n,
                 const size_t c,
                 const GRID_ET y_n,
                 const GRID_ET x_n,
                 const get_padded_fn_t<DATA_ET>& get_padded,
                 const denormalize_fn_t<GRID_ET>& denormalize) {
    const auto y_d = denormalize(y_n, data_shape[2]);
    const auto x_d = denormalize(x_n, data_shape[3]);
    const auto y_topleft = std::floor(y_d);
    const auto x_topleft = std::floor(x_d);
    const auto dy = y_d - y_topleft;
    const auto dx = x_d - x_topleft;
    const auto v00 = get_padded(data, data_shape, n, c, static_cast<long>(y_topleft), static_cast<long>(x_topleft));
    const auto v01 = get_padded(data, data_shape, n, c, static_cast<long>(y_topleft), static_cast<long>(x_topleft + 1));
    const auto v10 = get_padded(data, data_shape, n, c, static_cast<long>(y_topleft + 1), static_cast<long>(x_topleft));
    const auto v11 =
        get_padded(data, data_shape, n, c, static_cast<long>(y_topleft + 1), static_cast<long>(x_topleft + 1));

    const auto q0 = (1 - dx) * v00 + dx * v01;
    const auto q1 = (1 - dx) * v10 + dx * v11;
    return dy * q1 + (1 - dy) * q0;
}

template <typename DATA_ET, typename GRID_ET>
DATA_ET nearest(const DATA_ET* data,
                const Shape& data_shape,
                const size_t n,
                const size_t c,
                const GRID_ET y_n,
                const GRID_ET x_n,
                const get_padded_fn_t<DATA_ET>& get_padded,
                const denormalize_fn_t<GRID_ET>& denormalize) {
    const auto y_nearest = std::lrint(denormalize(y_n, data_shape[2]));
    const auto x_nearest = std::lrint(denormalize(x_n, data_shape[3]));
    return get_padded(data, data_shape, n, c, y_nearest, x_nearest);
}

template <typename T>
using vector_4_t = typename std::array<T, 4>;
template <typename T>
using matrix_4x4_t = typename std::array<vector_4_t<T>, 4>;

// formula taken from
// https://github.com/onnx/onnx/blob/a92870cdf359297495a118184dca2eaecee4b717/onnx/backend/test/case/node/resize.py#L201-L207
template <typename T>
vector_4_t<T> cubic_coeffs(const T r, const T A = -0.75) {
    vector_4_t<T> v;
    v[0] = ((A * (r + 1) - 5 * A) * (r + 1) + 8 * A) * (r + 1) - 4 * A;
    v[1] = ((A + 2) * r - (A + 3)) * r * r + 1;
    v[2] = ((A + 2) * (1 - r) - (A + 3)) * (1 - r) * (1 - r) + 1;
    v[3] = ((A * (2 - r) - 5 * A) * (2 - r) + 8 * A) * (2 - r) - 4 * A;
    return v;
}

template <typename DATA_ET>
matrix_4x4_t<DATA_ET> gather_4x4(const DATA_ET* data,
                                 const Shape& data_shape,
                                 const size_t n,
                                 const size_t c,
                                 const long y_topleft,
                                 const long x_topleft,
                                 const get_padded_fn_t<DATA_ET>& get_padded) {
    matrix_4x4_t<DATA_ET> s;
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i)
            s[j][i] = get_padded(data, data_shape, n, c, y_topleft + j, x_topleft + i);
    return s;
}

template <typename DATA_ET, typename GRID_ET>
DATA_ET bicubic(const DATA_ET* data,
                const Shape& data_shape,
                const size_t n,
                const size_t c,
                const GRID_ET y_n,
                const GRID_ET x_n,
                const get_padded_fn_t<DATA_ET>& get_padded,
                const denormalize_fn_t<GRID_ET>& denormalize) {
    const auto y_d = denormalize(y_n, data_shape[2]);
    const auto x_d = denormalize(x_n, data_shape[3]);
    const auto y_topleft = std::floor(y_d);
    const auto x_topleft = std::floor(x_d);
    const auto dy = y_d - y_topleft;
    const auto dx = x_d - x_topleft;
    const auto s = gather_4x4(data,
                              data_shape,
                              n,
                              c,
                              static_cast<long>(y_topleft - 1),
                              static_cast<long>(x_topleft - 1),
                              get_padded);

    const auto cy = cubic_coeffs(dy);
    const auto cx = cubic_coeffs(dx);
    vector_4_t<DATA_ET> p;
    std::transform(s.begin(), s.end(), p.begin(), [&cx](const vector_4_t<DATA_ET>& v) {
        return std::inner_product(cx.begin(), cx.end(), v.begin(), static_cast<DATA_ET>(0));
    });
    return std::inner_product(cy.begin(), cy.end(), p.begin(), static_cast<DATA_ET>(0));
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
    assert(data_shape[0] == grid_shape[0] && grid_shape[3] == 2);

    const auto N = data_shape[0];
    const auto C = data_shape[1];
    const auto H_out = grid_shape[1];
    const auto W_out = grid_shape[2];
    const Shape output_shape{N, C, H_out, W_out};

    const auto prev_rounding_mode = std::fegetround();
    std::fesetround(FE_TONEAREST);

    get_padded_fn_t<DATA_ET> get_padded_fn;
    switch (padding_mode) {
    default:
    case ov::op::v9::GridSample::PaddingMode::ZEROS:
        get_padded_fn = zeros_padding<DATA_ET>;
        break;
    case ov::op::v9::GridSample::PaddingMode::BORDER:
        get_padded_fn = border_padding<DATA_ET>;
        break;
    case ov::op::v9::GridSample::PaddingMode::REFLECTION:
        if (align_corners)
            get_padded_fn = reflection_data_with_align<DATA_ET>;
        else
            get_padded_fn = reflection_data_no_align<DATA_ET>;
        break;
    }

    denormalize_fn_t<GRID_ET> denormalize_fn;
    if (align_corners)
        denormalize_fn = rescale_align<GRID_ET>;
    else
        denormalize_fn = rescale_noalign<GRID_ET>;

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t y = 0; y < H_out; ++y) {
                for (size_t x = 0; x < W_out; ++x) {
                    const auto y_n = get_single_value(grid, grid_shape, index_4D_t{n, y, x, 1});
                    const auto x_n = get_single_value(grid, grid_shape, index_4D_t{n, y, x, 0});

                    auto& out = get_single_value(output, output_shape, index_4D_t{n, c, y, x});

                    switch (interpolation_mode) {
                    case ov::op::v9::GridSample::InterpolationMode::BILINEAR:
                        out = bilinear(data, data_shape, n, c, y_n, x_n, get_padded_fn, denormalize_fn);
                        break;
                    case ov::op::v9::GridSample::InterpolationMode::NEAREST:
                        out = nearest(data, data_shape, n, c, y_n, x_n, get_padded_fn, denormalize_fn);
                        break;
                    case ov::op::v9::GridSample::InterpolationMode::BICUBIC:
                        out = bicubic(data, data_shape, n, c, y_n, x_n, get_padded_fn, denormalize_fn);
                        break;
                    }
                }
            }
        }
    }

    std::fesetround(prev_rounding_mode);
}
}  // namespace reference
}  // namespace ov
