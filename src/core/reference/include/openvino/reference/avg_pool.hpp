// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cfenv>
#include <cmath>
#include <numeric>
#include <vector>

#include "openvino/core/axis_vector.hpp"
#include "openvino/core/coordinate.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/reference/rounding_guard.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {
namespace {
inline bool elem_in_padding_area(const Coordinate& kernel_position,
                                 const Coordinate& kernel_offset,
                                 const Shape& data_shape) {
    for (size_t dim = 0; dim + 2 < data_shape.size(); ++dim) {
        if (static_cast<int64_t>(kernel_position[dim]) + static_cast<int64_t>(kernel_offset[dim]) < 0LL ||
            kernel_position[dim] + kernel_offset[dim] >= data_shape[dim + 2]) {
            return true;
        }
    }

    return false;
}

inline Coordinate calculate_kernel_position(const Coordinate& out_elem_coord,
                                            const Strides& kernel_strides,
                                            const Shape& pads_begin) {
    Coordinate top_left_corner;
    top_left_corner.reserve(out_elem_coord.size());
    for (size_t i = 0u; i < out_elem_coord.size(); ++i) {
        top_left_corner.emplace_back(out_elem_coord[i] * kernel_strides[i] - pads_begin[i]);
    }
    return top_left_corner;
}

namespace kernel {
template <typename Values_t>
void avg_pool_3d(const Values_t* data,
                 Values_t* out,
                 const Shape& data_shape,
                 const Shape& out_shape,
                 const Shape& kernel,
                 const Strides& kernel_strides,
                 const Shape& pads_begin,
                 const Shape& pads_end,
                 const bool pads_in_avg) {
    // helper constants(axes) denoting dimensions in the input data shape and kernel shape
    constexpr size_t data_D = 2, data_H = 3, data_W = 4;
    constexpr size_t kernel_D = 0, kernel_H = 1, kernel_W = 2;

    // select max elem and its index for each "placeholder" in the out buffer (pointed to by out_idx)
    size_t out_idx = 0u;
    for (size_t out_channel = 0u; out_channel < out_shape[data_D]; ++out_channel) {
        for (size_t out_row = 0u; out_row < out_shape[data_H]; ++out_row) {
            for (size_t out_col = 0u; out_col < out_shape[data_W]; ++out_col) {
                auto sum = Values_t{0};
                auto count = size_t{0};

                const auto kernel_position =
                    calculate_kernel_position({out_channel, out_row, out_col}, kernel_strides, pads_begin);

                for (size_t kernel_channel = 0; kernel_channel < kernel[kernel_D]; ++kernel_channel) {
                    for (size_t kernel_row = 0; kernel_row < kernel[kernel_H]; ++kernel_row) {
                        for (size_t kernel_col = 0; kernel_col < kernel[kernel_W]; ++kernel_col) {
                            // offset from the top-left corner of the kernel for a given row and col
                            const Coordinate kernel_offset{kernel_channel, kernel_row, kernel_col};

                            const auto in_padding = elem_in_padding_area(kernel_position, kernel_offset, data_shape);
                            // ignore the elements in the padding area
                            if (!in_padding) {
                                // index of the flattened tensor element under the current row & column of the kernel
                                const size_t data_elem_index =
                                    data_shape[data_H] * data_shape[data_W] *
                                        (kernel_offset[kernel_D] + kernel_position[kernel_D]) +
                                    data_shape[data_W] * (kernel_offset[kernel_H] + kernel_position[kernel_H]) +
                                    kernel_offset[kernel_W] + kernel_position[kernel_W];

                                sum += data[data_elem_index];
                            }
                            if (pads_in_avg || !in_padding) {
                                ++count;
                            }
                        }
                    }
                }

                if (count != 0) {
                    if (std::is_same<Values_t, int8_t>::value || std::is_same<Values_t, uint8_t>::value) {
                        out[out_idx] = static_cast<Values_t>(std::nearbyint(sum / count));
                    } else {
                        out[out_idx] = sum / static_cast<Values_t>(count);
                    }
                } else {
                    out[out_idx] = Values_t{0};
                }
                ++out_idx;
            }
        }
    }
}
}  // namespace kernel
}  // namespace

template <typename T>
void avg_pool(const T* const arg,
              T* const out,
              const Shape& arg_shape,
              const Shape& out_shape,
              const Shape& window_shape,
              const Strides& window_movement_strides,
              const Shape& padding_below,
              const Shape& padding_above,
              const bool include_padding_in_avg_computation) {
    if (window_shape.size() > 3)
        return;
    const RoundingGuard rounding_g{FE_TONEAREST};

    const auto not_zero = [](size_t p) {
        return p != 0;
    };
    const auto pads_in_avg =
        include_padding_in_avg_computation && (std::any_of(padding_below.begin(), padding_below.end(), not_zero) ||
                                               std::any_of(padding_above.begin(), padding_above.end(), not_zero));

    Shape arg_shape_5D{arg_shape};
    Shape out_shape_5D{out_shape};
    Shape window_shape_3D{window_shape};
    Strides window_movement_strides_3D{window_movement_strides};
    Shape padding_below_3D{padding_below};
    Shape padding_above_3D{padding_above};

    if (window_shape.size() < 3) {
        const size_t dim_diff = 3 - window_shape.size();
        arg_shape_5D.insert(std::next(arg_shape_5D.begin(), 2), dim_diff, 1);
        out_shape_5D.insert(std::next(out_shape_5D.begin(), 2), dim_diff, 1);
        window_shape_3D.insert(window_shape_3D.begin(), dim_diff, 1);
        window_movement_strides_3D.insert(window_movement_strides_3D.begin(), dim_diff, 1);
        padding_below_3D.insert(padding_below_3D.begin(), dim_diff, 0);
        padding_above_3D.insert(padding_above_3D.begin(), dim_diff, 0);
    }

    const auto data_batch_elems = shape_size(std::begin(arg_shape) + 1, std::end(arg_shape));
    const auto data_channel_elems = shape_size(std::begin(arg_shape) + 2, std::end(arg_shape));

    const auto out_batch_elems = shape_size(std::begin(out_shape) + 1, std::end(out_shape));
    const auto out_channel_elems = shape_size(std::begin(out_shape) + 2, std::end(out_shape));

    for (size_t b = 0; b < arg_shape[0]; ++b) {
        for (size_t c = 0; c < arg_shape[1]; ++c) {
            const T* data_channel_first_elem = arg + b * data_batch_elems + c * data_channel_elems;
            T* out_channel_first_elem = out + b * out_batch_elems + c * out_channel_elems;
            kernel::avg_pool_3d(data_channel_first_elem,
                                out_channel_first_elem,
                                arg_shape_5D,
                                out_shape_5D,
                                window_shape_3D,
                                window_movement_strides_3D,
                                padding_below_3D,
                                padding_above_3D,
                                pads_in_avg);
        }
    }
}
}  // namespace reference
}  // namespace ov
