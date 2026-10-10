// Copyright (C) 2018-2026 Intel Corporation
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
#include "openvino/reference/rounding_guard.hpp"

namespace ov {
namespace reference {
namespace {

using index_4D_t = typename std::array<size_t, 4>;
using index_5D_t = typename std::array<size_t, 5>;

template <typename GRID_ET>
using denormalize_fn_t = GRID_ET (*)(GRID_ET, size_t);

template <typename DATA_ET>
using get_padded_fn_t = DATA_ET (*)(const DATA_ET*, const Shape&, size_t, size_t, long, long);

// 3D-spatial (volumetric) variant of the padded-access function: takes (z, y, x) integer coordinates.
template <typename DATA_ET>
using get_padded_3d_fn_t = DATA_ET (*)(const DATA_ET*, const Shape&, size_t, size_t, long, long, long);

template <typename DATA_ET, typename GRID_ET>
using interpolate_fn_t = DATA_ET (*)(const DATA_ET* data,
                                     const Shape&,
                                     const size_t n,
                                     const size_t c,
                                     const GRID_ET,
                                     const GRID_ET,
                                     const get_padded_fn_t<DATA_ET>&,
                                     const denormalize_fn_t<GRID_ET>&);

template <typename T, std::size_t N>
T& get_single_value(T* buffer, const Shape& shape, const std::array<size_t, N>& index) {
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
    const auto dy = static_cast<DATA_ET>(y_d - y_topleft);
    const auto dx = static_cast<DATA_ET>(x_d - x_topleft);
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

// ---------------------------------------------------------------------------
// 5D (volumetric) GridSample helpers.
//
// Data layout is [N, C, D, H, W]; the spatial axes are data_shape[2] (D),
// data_shape[3] (H) and data_shape[4] (W). The grid stores (x, y, z) triplets
// where x maps to W, y maps to H and z maps to D - mirroring the (x, y) order of
// the 4D path. Only `bilinear` (trilinear) and `nearest` are defined for 5D in
// both PyTorch and ONNX; `bicubic` is rejected by the operator validation and is
// therefore never dispatched here.
// ---------------------------------------------------------------------------

template <typename DATA_ET, typename GRID_ET>
using interpolate_3d_fn_t = DATA_ET (*)(const DATA_ET* data,
                                        const Shape&,
                                        const size_t n,
                                        const size_t c,
                                        const GRID_ET,
                                        const GRID_ET,
                                        const GRID_ET,
                                        const get_padded_3d_fn_t<DATA_ET>&,
                                        const denormalize_fn_t<GRID_ET>&);

template <typename DATA_ET>
DATA_ET zeros_padding_3d(const DATA_ET* data,
                         const Shape& data_shape,
                         const size_t n,
                         const size_t c,
                         const long z_d,
                         const long y_d,
                         const long x_d) {
    const auto D = static_cast<long>(data_shape[2]);
    const auto H = static_cast<long>(data_shape[3]);
    const auto W = static_cast<long>(data_shape[4]);
    if (z_d < 0 || y_d < 0 || x_d < 0 || z_d >= D || y_d >= H || x_d >= W) {
        return 0;
    } else {
        return get_single_value(data,
                                data_shape,
                                index_5D_t{n, c, static_cast<size_t>(z_d), static_cast<size_t>(y_d),
                                           static_cast<size_t>(x_d)});
    }
}

template <typename DATA_ET>
DATA_ET border_padding_3d(const DATA_ET* data,
                          const Shape& data_shape,
                          const size_t n,
                          const size_t c,
                          const long z_d,
                          const long y_d,
                          const long x_d) {
    const auto D = static_cast<long>(data_shape[2]);
    const auto H = static_cast<long>(data_shape[3]);
    const auto W = static_cast<long>(data_shape[4]);
    const auto z = static_cast<size_t>(std::min(std::max(z_d, 0l), D - 1));
    const auto y = static_cast<size_t>(std::min(std::max(y_d, 0l), H - 1));
    const auto x = static_cast<size_t>(std::min(std::max(x_d, 0l), W - 1));
    return get_single_value(data, data_shape, index_5D_t{n, c, z, y, x});
}

// Reflects a single axis coordinate using the same arithmetic as the 4D path.
inline long reflect_index_no_align(long coord, const long size) {
    const auto size_2 = size * 2l;
    coord = (coord % size_2 + size_2) % size_2;
    return coord >= size ? size_2 - 1 - coord : coord;
}

inline long reflect_index_with_align(long coord, const long size) {
    const auto span = size == 1 ? 1 : 2 * (size - 1);
    coord = std::abs(coord) % span;
    return coord >= size ? span - coord : coord;
}

template <typename DATA_ET>
DATA_ET reflection_data_no_align_3d(const DATA_ET* data,
                                    const Shape& data_shape,
                                    const size_t n,
                                    const size_t c,
                                    long z_d,
                                    long y_d,
                                    long x_d) {
    const auto z = static_cast<size_t>(reflect_index_no_align(z_d, static_cast<long>(data_shape[2])));
    const auto y = static_cast<size_t>(reflect_index_no_align(y_d, static_cast<long>(data_shape[3])));
    const auto x = static_cast<size_t>(reflect_index_no_align(x_d, static_cast<long>(data_shape[4])));
    return get_single_value(data, data_shape, index_5D_t{n, c, z, y, x});
}

template <typename DATA_ET>
DATA_ET reflection_data_with_align_3d(const DATA_ET* data,
                                      const Shape& data_shape,
                                      const size_t n,
                                      const size_t c,
                                      long z_d,
                                      long y_d,
                                      long x_d) {
    const auto z = static_cast<size_t>(reflect_index_with_align(z_d, static_cast<long>(data_shape[2])));
    const auto y = static_cast<size_t>(reflect_index_with_align(y_d, static_cast<long>(data_shape[3])));
    const auto x = static_cast<size_t>(reflect_index_with_align(x_d, static_cast<long>(data_shape[4])));
    return get_single_value(data, data_shape, index_5D_t{n, c, z, y, x});
}

template <typename DATA_ET, typename GRID_ET>
DATA_ET trilinear(const DATA_ET* data,
                  const Shape& data_shape,
                  const size_t n,
                  const size_t c,
                  const GRID_ET z_n,
                  const GRID_ET y_n,
                  const GRID_ET x_n,
                  const get_padded_3d_fn_t<DATA_ET>& get_padded,
                  const denormalize_fn_t<GRID_ET>& denormalize) {
    const auto z_d = denormalize(z_n, data_shape[2]);
    const auto y_d = denormalize(y_n, data_shape[3]);
    const auto x_d = denormalize(x_n, data_shape[4]);
    const auto z_topleft = std::floor(z_d);
    const auto y_topleft = std::floor(y_d);
    const auto x_topleft = std::floor(x_d);
    const auto dz = static_cast<DATA_ET>(z_d - z_topleft);
    const auto dy = static_cast<DATA_ET>(y_d - y_topleft);
    const auto dx = static_cast<DATA_ET>(x_d - x_topleft);

    const auto z0 = static_cast<long>(z_topleft);
    const auto y0 = static_cast<long>(y_topleft);
    const auto x0 = static_cast<long>(x_topleft);

    const auto v000 = get_padded(data, data_shape, n, c, z0, y0, x0);
    const auto v001 = get_padded(data, data_shape, n, c, z0, y0, x0 + 1);
    const auto v010 = get_padded(data, data_shape, n, c, z0, y0 + 1, x0);
    const auto v011 = get_padded(data, data_shape, n, c, z0, y0 + 1, x0 + 1);
    const auto v100 = get_padded(data, data_shape, n, c, z0 + 1, y0, x0);
    const auto v101 = get_padded(data, data_shape, n, c, z0 + 1, y0, x0 + 1);
    const auto v110 = get_padded(data, data_shape, n, c, z0 + 1, y0 + 1, x0);
    const auto v111 = get_padded(data, data_shape, n, c, z0 + 1, y0 + 1, x0 + 1);

    // Interpolate along x, then y, then z.
    const auto c00 = (1 - dx) * v000 + dx * v001;
    const auto c01 = (1 - dx) * v010 + dx * v011;
    const auto c10 = (1 - dx) * v100 + dx * v101;
    const auto c11 = (1 - dx) * v110 + dx * v111;

    const auto c0 = (1 - dy) * c00 + dy * c01;
    const auto c1 = (1 - dy) * c10 + dy * c11;

    return (1 - dz) * c0 + dz * c1;
}

template <typename DATA_ET, typename GRID_ET>
DATA_ET nearest_3d(const DATA_ET* data,
                   const Shape& data_shape,
                   const size_t n,
                   const size_t c,
                   const GRID_ET z_n,
                   const GRID_ET y_n,
                   const GRID_ET x_n,
                   const get_padded_3d_fn_t<DATA_ET>& get_padded,
                   const denormalize_fn_t<GRID_ET>& denormalize) {
    const auto z_nearest = std::lrint(denormalize(z_n, data_shape[2]));
    const auto y_nearest = std::lrint(denormalize(y_n, data_shape[3]));
    const auto x_nearest = std::lrint(denormalize(x_n, data_shape[4]));
    return get_padded(data, data_shape, n, c, z_nearest, y_nearest, x_nearest);
}
}  // namespace

template <typename DATA_ET, typename GRID_ET>
void grid_sample_2d_spatial(DATA_ET* output,
                            const DATA_ET* data,
                            const GRID_ET* grid,
                            const Shape& data_shape,
                            const Shape& grid_shape,
                            const bool align_corners,
                            const ov::op::v9::GridSample::InterpolationMode interpolation_mode,
                            const ov::op::v9::GridSample::PaddingMode padding_mode) {
    const auto N = data_shape[0];
    const auto C = data_shape[1];
    const auto H_out = grid_shape[1];
    const auto W_out = grid_shape[2];
    const Shape output_shape{N, C, H_out, W_out};

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
        get_padded_fn = align_corners ? reflection_data_with_align<DATA_ET> : reflection_data_no_align<DATA_ET>;
        break;
    }

    const auto denormalize_fn = align_corners ? rescale_align<GRID_ET> : rescale_noalign<GRID_ET>;

    interpolate_fn_t<DATA_ET, GRID_ET> interpolate_fn;
    switch (interpolation_mode) {
    default:
    case ov::op::v9::GridSample::InterpolationMode::BILINEAR:
        interpolate_fn = bilinear<DATA_ET, GRID_ET>;
        break;
    case ov::op::v9::GridSample::InterpolationMode::NEAREST:
        interpolate_fn = nearest<DATA_ET, GRID_ET>;
        break;
    case ov::op::v9::GridSample::InterpolationMode::BICUBIC:
        interpolate_fn = bicubic<DATA_ET, GRID_ET>;
        break;
    }

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t y = 0; y < H_out; ++y) {
                for (size_t x = 0; x < W_out; ++x) {
                    const auto y_n = get_single_value(grid, grid_shape, index_4D_t{n, y, x, 1});
                    const auto x_n = get_single_value(grid, grid_shape, index_4D_t{n, y, x, 0});

                    auto& out = get_single_value(output, output_shape, index_4D_t{n, c, y, x});
                    out = interpolate_fn(data, data_shape, n, c, y_n, x_n, get_padded_fn, denormalize_fn);
                }
            }
        }
    }
}

template <typename DATA_ET, typename GRID_ET>
void grid_sample_3d_spatial(DATA_ET* output,
                            const DATA_ET* data,
                            const GRID_ET* grid,
                            const Shape& data_shape,
                            const Shape& grid_shape,
                            const bool align_corners,
                            const ov::op::v9::GridSample::InterpolationMode interpolation_mode,
                            const ov::op::v9::GridSample::PaddingMode padding_mode) {
    const auto N = data_shape[0];
    const auto C = data_shape[1];
    const auto D_out = grid_shape[1];
    const auto H_out = grid_shape[2];
    const auto W_out = grid_shape[3];
    const Shape output_shape{N, C, D_out, H_out, W_out};

    get_padded_3d_fn_t<DATA_ET> get_padded_fn;
    switch (padding_mode) {
    default:
    case ov::op::v9::GridSample::PaddingMode::ZEROS:
        get_padded_fn = zeros_padding_3d<DATA_ET>;
        break;
    case ov::op::v9::GridSample::PaddingMode::BORDER:
        get_padded_fn = border_padding_3d<DATA_ET>;
        break;
    case ov::op::v9::GridSample::PaddingMode::REFLECTION:
        get_padded_fn = align_corners ? reflection_data_with_align_3d<DATA_ET> : reflection_data_no_align_3d<DATA_ET>;
        break;
    }

    const auto denormalize_fn = align_corners ? rescale_align<GRID_ET> : rescale_noalign<GRID_ET>;

    // `bicubic` is undefined for volumetric input and is rejected by the operator validation,
    // so only `bilinear` (trilinear) and `nearest` reach this point.
    interpolate_3d_fn_t<DATA_ET, GRID_ET> interpolate_fn;
    switch (interpolation_mode) {
    default:
    case ov::op::v9::GridSample::InterpolationMode::BILINEAR:
        interpolate_fn = trilinear<DATA_ET, GRID_ET>;
        break;
    case ov::op::v9::GridSample::InterpolationMode::NEAREST:
        interpolate_fn = nearest_3d<DATA_ET, GRID_ET>;
        break;
    }

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t z = 0; z < D_out; ++z) {
                for (size_t y = 0; y < H_out; ++y) {
                    for (size_t x = 0; x < W_out; ++x) {
                        const auto x_n = get_single_value(grid, grid_shape, index_5D_t{n, z, y, x, 0});
                        const auto y_n = get_single_value(grid, grid_shape, index_5D_t{n, z, y, x, 1});
                        const auto z_n = get_single_value(grid, grid_shape, index_5D_t{n, z, y, x, 2});

                        auto& out = get_single_value(output, output_shape, index_5D_t{n, c, z, y, x});
                        out = interpolate_fn(data, data_shape, n, c, z_n, y_n, x_n, get_padded_fn, denormalize_fn);
                    }
                }
            }
        }
    }
}

template <typename DATA_ET, typename GRID_ET>
void grid_sample(DATA_ET* output,
                 const DATA_ET* data,
                 const GRID_ET* grid,
                 const Shape& data_shape,
                 const Shape& grid_shape,
                 const bool align_corners,
                 const ov::op::v9::GridSample::InterpolationMode interpolation_mode,
                 const ov::op::v9::GridSample::PaddingMode padding_mode) {
    assert(data_shape.size() == grid_shape.size());
    assert(data_shape[0] == grid_shape[0]);

    const RoundingGuard rounding_guard{FE_TONEAREST};

    if (data_shape.size() == 5) {
        assert(grid_shape[4] == 3);
        grid_sample_3d_spatial(output,
                               data,
                               grid,
                               data_shape,
                               grid_shape,
                               align_corners,
                               interpolation_mode,
                               padding_mode);
    } else {
        assert(data_shape.size() == 4 && grid_shape[3] == 2);
        grid_sample_2d_spatial(output,
                               data,
                               grid,
                               data_shape,
                               grid_shape,
                               align_corners,
                               interpolation_mode,
                               padding_mode);
    }
}
}  // namespace reference
}  // namespace ov
