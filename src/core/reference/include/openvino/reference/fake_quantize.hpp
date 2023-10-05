// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {
namespace fake_quantize_details {
template <typename T>
inline T quantize(const T& arg,
                  const T& in_low,
                  const T& in_high,
                  const T& out_low,
                  const T& out_high,
                  const size_t& levels) {
    if (arg <= std::min(in_low, in_high)) {
        return out_low;
    } else if (arg > std::max(in_low, in_high)) {
        return out_high;
    }
    return static_cast<T>(std::nearbyint((arg - in_low) / (in_high - in_low) * (levels - 1)) / (levels - 1) *
                              (out_high - out_low) +
                          out_low);
}

}  // namespace fake_quantize_details

template <typename T>
void fake_quantize(const T* const arg,
                   const T* const in_low,
                   const T* const in_high,
                   const T* const out_low,
                   const T* const out_high,
                   T* const out,
                   const Shape& arg_shape,
                   const Shape& in_low_shape,
                   const Shape& in_high_shape,
                   const Shape& out_low_shape,
                   const Shape& out_high_shape,
                   size_t levels,
                   const op::AutoBroadcastSpec& broadcast) {
    using namespace fake_quantize_details;

    if (shape_size(in_low_shape) == 1 && shape_size(in_high_shape) == 1 && shape_size(out_low_shape) == 1 &&
        shape_size(out_high_shape) == 1) {
        const size_t arg_size = shape_size(arg_shape);
        const auto q = [=](const T& a) {
            return quantize(a, *in_low, *in_high, *out_low, *out_high, levels);
        };
        for (size_t i = 0; i < arg_size; ++i) {
            out[i] = q(arg[i]);
        }
    } else {
        OPENVINO_ASSERT(in_low_shape.size() <= arg_shape.size() && in_high_shape.size() <= arg_shape.size() &&
                            out_low_shape.size() <= arg_shape.size() && out_high_shape.size() <= arg_shape.size(),
                        "Tensors with input\\output ranges should have rank less or "
                        "equal to data tensor rank equal to ",
                        arg_shape.size());

        Shape arg0_padded_shape = arg_shape;
        Shape arg1_padded_shape = in_low_shape;
        Shape arg2_padded_shape = in_high_shape;
        Shape arg3_padded_shape = out_low_shape;
        Shape arg4_padded_shape = out_high_shape;

        size_t max_shape_size = arg_shape.size();

        while (arg0_padded_shape.size() < max_shape_size) {
            arg0_padded_shape.insert(arg0_padded_shape.begin(), 1);
        }

        while (arg1_padded_shape.size() < max_shape_size) {
            arg1_padded_shape.insert(arg1_padded_shape.begin(), 1);
        }

        while (arg2_padded_shape.size() < max_shape_size) {
            arg2_padded_shape.insert(arg2_padded_shape.begin(), 1);
        }

        while (arg3_padded_shape.size() < max_shape_size) {
            arg3_padded_shape.insert(arg3_padded_shape.begin(), 1);
        }

        while (arg4_padded_shape.size() < max_shape_size) {
            arg4_padded_shape.insert(arg4_padded_shape.begin(), 1);
        }

        Shape arg0_squeezed_shape, arg1_squeezed_shape, arg2_squeezed_shape, arg3_squeezed_shape, arg4_squeezed_shape;
        AxisSet arg0_squeezed_axes, arg1_squeezed_axes, arg2_squeezed_axes, arg3_squeezed_axes, arg4_squeezed_axes;
        Shape output_shape;

        for (size_t i = 0; i < max_shape_size; i++) {
            if (arg1_padded_shape[i] == 1) {
                arg1_squeezed_axes.insert(i);
            } else {
                arg1_squeezed_shape.push_back(arg1_padded_shape[i]);
            }

            if (arg2_padded_shape[i] == 1) {
                arg2_squeezed_axes.insert(i);
            } else {
                arg2_squeezed_shape.push_back(arg2_padded_shape[i]);
            }

            if (arg0_padded_shape[i] == 1) {
                arg0_squeezed_axes.insert(i);
            } else {
                arg0_squeezed_shape.push_back(arg0_padded_shape[i]);
            }

            if (arg3_padded_shape[i] == 1) {
                arg3_squeezed_axes.insert(i);
            } else {
                arg3_squeezed_shape.push_back(arg3_padded_shape[i]);
            }

            if (arg4_padded_shape[i] == 1) {
                arg4_squeezed_axes.insert(i);
            } else {
                arg4_squeezed_shape.push_back(arg4_padded_shape[i]);
            }

            output_shape.push_back(std::max({arg0_padded_shape[i],
                                             arg2_padded_shape[i],
                                             arg1_padded_shape[i],
                                             arg3_padded_shape[i],
                                             arg4_padded_shape[i]}));
        }

        CoordinateTransformBasic arg0_transform(arg0_squeezed_shape);
        CoordinateTransformBasic arg1_transform(arg1_squeezed_shape);
        CoordinateTransformBasic arg2_transform(arg2_squeezed_shape);
        CoordinateTransformBasic arg3_transform(arg3_squeezed_shape);
        CoordinateTransformBasic arg4_transform(arg4_squeezed_shape);
        CoordinateTransformBasic output_transform(output_shape);

        const auto arg0_strides = row_major_strides(arg0_squeezed_shape);
        const auto arg1_strides = row_major_strides(arg1_squeezed_shape);
        const auto arg2_strides = row_major_strides(arg2_squeezed_shape);
        const auto arg3_strides = row_major_strides(arg3_squeezed_shape);
        const auto arg4_strides = row_major_strides(arg4_squeezed_shape);
        const auto output_strides = row_major_strides(output_shape);

        for (const Coordinate& output_coord : output_transform) {
            const auto arg0_coord = util::reduce(output_coord, arg0_squeezed_axes);
            const auto arg1_coord = util::reduce(output_coord, arg1_squeezed_axes);
            const auto arg2_coord = util::reduce(output_coord, arg2_squeezed_axes);
            const auto arg3_coord = util::reduce(output_coord, arg3_squeezed_axes);
            const auto arg4_coord = util::reduce(output_coord, arg4_squeezed_axes);

            const size_t arg0_idx =
                std::inner_product(arg0_coord.begin(), arg0_coord.end(), arg0_strides.begin(), uint64_t(0));
            const size_t arg1_idx =
                std::inner_product(arg1_coord.begin(), arg1_coord.end(), arg1_strides.begin(), uint64_t(0));
            const size_t arg2_idx =
                std::inner_product(arg2_coord.begin(), arg2_coord.end(), arg2_strides.begin(), uint64_t(0));
            const size_t arg3_idx =
                std::inner_product(arg3_coord.begin(), arg3_coord.end(), arg3_strides.begin(), uint64_t(0));
            const size_t arg4_idx =
                std::inner_product(arg4_coord.begin(), arg4_coord.end(), arg4_strides.begin(), uint64_t(0));
            const size_t output_idx =
                std::inner_product(output_coord.begin(), output_coord.end(), output_strides.begin(), uint64_t(0));
            out[output_idx] = quantize(arg[arg0_idx],
                                       in_low[arg1_idx],
                                       in_high[arg2_idx],
                                       out_low[arg3_idx],
                                       out_high[arg4_idx],
                                       levels);
        }
    }
}
}  // namespace reference
}  // namespace ov
