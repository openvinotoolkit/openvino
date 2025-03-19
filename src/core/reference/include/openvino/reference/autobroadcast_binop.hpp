// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <numeric>
#include <utility>

#include "openvino/core/shape_util.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/reference/utils/coordinate_index.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {
namespace internal {
inline void row_major_strides(const Shape& shape, size_t* strides, size_t size) noexcept {
    size_t* st = strides + size - 1;
    size_t s = 1;
    for (auto d = shape.rbegin(), last = shape.rend(); d != last; ++d) {
        *st-- = s;
        s *= *d;
    }
    std::fill(strides, st + 1, s);
}

template <typename C, typename T>
inline T value_with_padding_or(const C& arr, size_t padding, size_t idx, T&& default_value) {
    return idx < padding ? std::forward<T>(default_value) : static_cast<T>(arr[idx - padding]);
}

template <int A0, int A1, typename T, typename U, typename Functor>
inline void numpy_autobroadcast_binop(const T* arg0,
                                      const T* arg1,
                                      U* out,
                                      const Shape& shape0,
                                      const Shape& shape1,
                                      const size_t* strides0,
                                      const size_t* strides1,
                                      const size_t padding0,
                                      const size_t padding1,
                                      const Shape& output_shape,
                                      const size_t axis,
                                      const size_t stride,
                                      Functor&& elementwise_functor) {
    for (CoordinateIterator it(output_shape), ite = CoordinateIterator::end();;) {
        for (size_t i = 0; i < stride; ++i, ++out) {
            *out = elementwise_functor(arg0[i * A0], arg1[i * A1]);
        }

        arg0 += A0 ? stride : 1;
        arg1 += A1 ? stride : 1;

        auto p = it.advance(axis);

        if (it == ite)
            break;

        if (value_with_padding_or(shape0, padding0, p, 1) == 1)
            arg0 -= strides0[p];

        if (value_with_padding_or(shape1, padding1, p, 1) == 1)
            arg1 -= strides1[p];
    }
}

inline size_t calculate_fixed_axis(size_t axis, const size_t* strides) {
    while (axis > 0 && strides[axis - 1] == 1)
        --axis;
    return axis;
}
}  // namespace internal

/**
 * @brief Apply elementwise function for 2 inputs of same size.
 *
 * @param arg0  Pointer to input 0 data.
 * @param arg1  Pointer to input 1 data.
 * @param out   Pointer to output data.
 * @param count Number of elements in inputs
 * @param f     Binary elementwise functions.
 */
template <typename T, typename U, class Functor>
void no_broadcast_binop(const T* arg0, const T* arg1, U* out, const size_t count, Functor f) {
    for (auto last = arg0 + count; arg0 != last; ++arg0, ++arg1, ++out) {
        *out = f(*arg0, *arg1);
    }
}

/**
 * @brief Apply elementwise function for 2 inputs and apply NUMPY broadcasting.
 *
 * @param arg0       Pointer to input 0 data.
 * @param arg1       Pointer to input 1 data.
 * @param out        Pointer to output data.
 * @param arg0_shape Shape of input 0.
 * @param arg1_shape Shape of input 1.
 * @param f          Binary elementwise functions.
 */
template <typename T, typename U, class Functor>
void numpy_broadcast_binop(const T* arg0,
                           const T* arg1,
                           U* out,
                           const Shape& arg0_shape,
                           const Shape& arg1_shape,
                           Functor f) {
    // We'll be using CoordinateTransformBasic to handle the broadcasting. The general procedure is as follows:
    //
    // (1) Left pad the shorter of the two shapes with ones.
    // (2) Squeeze (remove ones from) both shapes, and record the squeezed axis indices.
    // (3) Using CoordinateTransformBasic, broadcast both args to the final output shape. The "broadcasted axes" will be
    //     those that were squeezed in step 2.
    //
    // Example:
    //
    //    Input shape->Padded shape->Squeezed Shape/Squeezed Axes
    //    -----------  ------------  ----------------------------
    // a: [ 3, 2, 1]   [ 3, 2, 1]    [ 3, 2   ]     {2}
    // b: [    1, 6]   [ 1, 1, 6]    [       6]     {0,1}
    //                   |  |  |
    //                   v  v  v
    //                 Output shape
    //                 ------------
    //                 [ 3, 2, 6]
    using namespace internal;

    size_t const shape_rank = std::max(arg0_shape.size(), arg1_shape.size()) + 1;

    // TODO: Use compiler-specific alloca() or variable-length array
    std::vector<size_t> tmp(shape_rank * 2);

    size_t* strides0 = tmp.data();
    size_t* strides1 = tmp.data() + shape_rank;

    row_major_strides(arg0_shape, strides0, shape_rank);
    row_major_strides(arg1_shape, strides1, shape_rank);

    size_t const padding0 = shape_rank - arg0_shape.size();
    size_t const padding1 = shape_rank - arg1_shape.size();

    Shape output_shape(shape_rank, 0);

    size_t axis = 0;

    for (size_t i = 0; i < shape_rank; ++i) {
        auto const dim0 = value_with_padding_or(arg0_shape, padding0, i, 1);
        auto const dim1 = value_with_padding_or(arg1_shape, padding1, i, 1);

        output_shape[i] = std::max(dim0, dim1);

        if (dim0 != dim1)
            axis = std::max(axis, i);
    }

    if (axis == 0) {
        no_broadcast_binop(arg0, arg1, out, strides0[0], f);
    } else if (strides0[axis] == 1 && value_with_padding_or(arg0_shape, padding0, axis, 1) == 1) {
        axis = calculate_fixed_axis(axis, strides0);

        internal::numpy_autobroadcast_binop<0, 1>(arg0,
                                                  arg1,
                                                  out,
                                                  arg0_shape,
                                                  arg1_shape,
                                                  strides0,
                                                  strides1,
                                                  padding0,
                                                  padding1,
                                                  output_shape,
                                                  axis,
                                                  strides1[axis],
                                                  f);
    } else if (strides1[axis] == 1 && value_with_padding_or(arg1_shape, padding1, axis, 1) == 1) {
        axis = calculate_fixed_axis(axis, strides1);

        internal::numpy_autobroadcast_binop<1, 0>(arg0,
                                                  arg1,
                                                  out,
                                                  arg0_shape,
                                                  arg1_shape,
                                                  strides0,
                                                  strides1,
                                                  padding0,
                                                  padding1,
                                                  output_shape,
                                                  axis,
                                                  strides0[axis],
                                                  f);
    } else
        internal::numpy_autobroadcast_binop<1, 1>(arg0,
                                                  arg1,
                                                  out,
                                                  arg0_shape,
                                                  arg1_shape,
                                                  strides0,
                                                  strides1,
                                                  padding0,
                                                  padding1,
                                                  output_shape,
                                                  axis,
                                                  strides0[axis],
                                                  f);
}

/**
 * @brief Apply elementwise function for 2 inputs and apply PDPP broadcasting.
 *
 * @param arg0       Pointer to input 0 data.
 * @param arg1       Pointer to input 1 data.
 * @param out        Pointer to output data.
 * @param arg0_shape Shape of input 0.
 * @param arg1_shape Shape of input 1.
 * @param axis       Start dimension index for broadcast arg1 shape into arg0.
 * @param f          Binary elementwise functions.
 */
template <typename T, typename U, class Functor>
void pdpd_broadcast_binop(const T* arg0,
                          const T* arg1,
                          U* out,
                          const Shape& arg0_shape,
                          const Shape& arg1_shape,
                          int64_t axis,
                          Functor f) {
    // We'll be using CoordinateTransformBasic to handle the broadcasting. No need to process arg0 and output shape will
    // be the same as arg0. We need to process arg1 and the general procedure is as follows:
    //
    // (1) Trim trailing ones from arg1 shape.
    // (2) Left and right pad arg1 to match arg0 shape. Axis is the index start to align between arg0 and arg1.
    // (3) Squeeze (remove ones from) arg1 shape, and record the squeezed axis indices.
    // (3) Using CoordinateTransformBasic, broadcast arg1 to the final output shape. The "broadcasted axes" will be
    //     those that were squeezed in step 23.
    //
    // Example:
    //
    //    Input shape->   Padded shape->   Squeezed Shape/Squeezed Axes
    //    -----------  ------------  ----------------------------
    // a: [ 3, 4, 5, 6]   [ 3, 4, 5, 6]    [ 3, 4, 5, 6]
    // b: [    4, 5,  ]   [ 1, 4, 5, 1]    [    4, 5   ]     {0,3}
    //                      |  |  |
    //                      v  v  v
    //                     Output shape
    //                     ------------
    //                    [ 3, 4, 5, 6]

    if (axis == -1) {
        axis = arg0_shape.size() - arg1_shape.size();
    }

    Shape arg1_padded_shape = arg1_shape;
    // Trim trailing ones
    while (arg1_padded_shape.size() > 0 && arg1_padded_shape.back() == 1) {
        arg1_padded_shape.pop_back();
    }

    for (int64_t i = 0; i < axis; ++i) {
        arg1_padded_shape.insert(arg1_padded_shape.begin(), 1);
    }

    while (arg1_padded_shape.size() < arg0_shape.size()) {
        arg1_padded_shape.insert(arg1_padded_shape.end(), 1);
    }

    Shape arg1_squeezed_shape;
    AxisSet arg1_squeezed_axes;

    for (size_t i = 0, size = arg0_shape.size(); i < size; i++) {
        if (arg1_padded_shape[i] == 1) {
            arg1_squeezed_axes.insert(i);
        } else {
            arg1_squeezed_shape.push_back(arg1_padded_shape[i]);
        }
    }

    const CoordinateTransformBasic output_transform{arg0_shape};

    for (const Coordinate& output_coord : output_transform) {
        const auto arg1_coord = util::reduce(output_coord, arg1_squeezed_axes);
        const auto out_index = coordinate_index(output_coord, arg0_shape);
        const auto arg0_index = coordinate_index(output_coord, arg0_shape);
        const auto arg1_index = coordinate_index(arg1_coord, arg1_squeezed_shape);
        out[out_index] = f(arg0[arg0_index], arg1[arg1_index]);
    }
}

/**
 * @brief Helper function to implement auto broadcasting elementwise binop references.
 *
 * @tparam T Element type of the input tensors.
 * @tparam U Element type of the output tensor.
 * @tparam Functor Type of the functor for the elementwise operation. Must support operator()(T,T), and operator()(T,T)
 *         must return a value of type U.
 *
 * @param arg0 Pointer to the buffer for left operand input tensor.
 * @param arg1 Pointer to the buffer for right operand input tensor.
 * @param out Pointer to the buffer for output tensor. This must be pre-allocated by the caller, and must be large
 *            enough to hold a tensor of the correct shape.
 * @param broadcast_spec Specification of the auto-broadcasting scheme.
 * @param elementwise_functor Functor implementing the elementwise operation to be applied across the input tensors.
 *                            Must accept two arguments of type T, and return a value of type U.
 */
template <typename T, typename U, typename Functor>
void autobroadcast_binop(const T* arg0,
                         const T* arg1,
                         U* out,
                         const Shape& arg0_shape,
                         const Shape& arg1_shape,
                         const op::AutoBroadcastSpec& broadcast_spec,
                         Functor elementwise_functor) {
    switch (broadcast_spec.m_type) {
    case op::AutoBroadcastType::NONE:
        no_broadcast_binop(arg0, arg1, out, shape_size(arg0_shape), elementwise_functor);
        break;
    case op::AutoBroadcastType::NUMPY:
        numpy_broadcast_binop(arg0, arg1, out, arg0_shape, arg1_shape, elementwise_functor);
        break;
    case op::AutoBroadcastType::PDPD:
        pdpd_broadcast_binop(arg0, arg1, out, arg0_shape, arg1_shape, broadcast_spec.m_axis, elementwise_functor);
        break;
    default:
        break;
    }
}

/**
 *
 * \brief Helper function to implement auto broadcasting elementwise ternary op references.
 * \tparam U Element type of the selector tensor.
 * \tparam T Element type of the input tensors.
 * \tparam Functor Type of the functor for the elementwise operation. Must support operator()(U,T,T), and
 * operator()(U,T,T) must return a value of type T.
 *
 * \param arg0 Pointer to the buffer for selector tensor.
 * \param arg1 Pointer to the buffer for left operand input tensor.
 * \param arg2 Pointer to the buffer for right operand input tensor.
 * \param out  Pointer to the buffer for output tensor. This must be pre-allocated by the caller, and must be large
 *             enough to hold a tensor of the correct shape.
 * \param broadcast_spec Specification of the auto-broadcasting scheme.
 * \param elementwise_functor Functor implementing the elementwise operation to be applied across the input tensors Must
 *                            accept an argument of type U and two of type T, and return a value of type T.
 */
template <typename T, typename U, typename Functor>
void autobroadcast_select(const U* arg0,
                          const T* arg1,
                          const T* arg2,
                          T* out,
                          const Shape& arg0_shape,
                          const Shape& arg1_shape,
                          const Shape& arg2_shape,
                          const op::AutoBroadcastSpec& broadcast_spec,
                          Functor elementwise_functor) {
    switch (broadcast_spec.m_type) {
    case op::AutoBroadcastType::NONE:
        for (auto last = arg0 + shape_size(arg0_shape); arg0 != last; ++arg0, ++arg1, ++arg2, ++out) {
            *out = elementwise_functor(*arg0, *arg1, *arg2);
        }
        break;
    case op::AutoBroadcastType::NUMPY:
        // Uses same approach as autobroadcast_binop.
        {
            Shape arg0_padded_shape = arg0_shape;
            Shape arg1_padded_shape = arg1_shape;
            Shape arg2_padded_shape = arg2_shape;

            size_t max_shape_size =
                std::max({arg0_padded_shape.size(), arg1_padded_shape.size(), arg2_padded_shape.size()});

            while (arg0_padded_shape.size() < max_shape_size) {
                arg0_padded_shape.insert(arg0_padded_shape.begin(), 1);
            }

            while (arg1_padded_shape.size() < max_shape_size) {
                arg1_padded_shape.insert(arg1_padded_shape.begin(), 1);
            }

            while (arg2_padded_shape.size() < max_shape_size) {
                arg2_padded_shape.insert(arg2_padded_shape.begin(), 1);
            }

            Shape arg0_squeezed_shape;
            Shape arg1_squeezed_shape;
            Shape arg2_squeezed_shape;
            AxisSet arg0_squeezed_axes;
            AxisSet arg1_squeezed_axes;
            AxisSet arg2_squeezed_axes;
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

                output_shape.push_back(std::max({arg0_padded_shape[i], arg2_padded_shape[i], arg1_padded_shape[i]}));
            }

            const CoordinateTransformBasic output_transform{output_shape};

            const auto arg0_strides = row_major_strides(arg0_squeezed_shape);
            const auto arg1_strides = row_major_strides(arg1_squeezed_shape);
            const auto arg2_strides = row_major_strides(arg2_squeezed_shape);
            const auto output_strides = row_major_strides(output_shape);

            for (const Coordinate& output_coord : output_transform) {
                const auto arg0_coord = util::reduce(output_coord, arg0_squeezed_axes);
                const auto arg1_coord = util::reduce(output_coord, arg1_squeezed_axes);
                const auto arg2_coord = util::reduce(output_coord, arg2_squeezed_axes);

                const size_t arg0_idx = coordinate_offset(arg0_coord, arg0_strides);
                const size_t arg1_idx = coordinate_offset(arg1_coord, arg1_strides);
                const size_t arg2_idx = coordinate_offset(arg2_coord, arg2_strides);
                const size_t output_idx = coordinate_offset(output_coord, output_strides);
                out[output_idx] = elementwise_functor(arg0[arg0_idx], arg1[arg1_idx], arg2[arg2_idx]);
            }
        }
        break;
    case op::AutoBroadcastType::PDPD: {
        // arg0 and arg2 are broadcast to arg1 shape
        int64_t axis = broadcast_spec.m_axis;
        if (axis == -1) {
            axis = arg1_shape.size() - arg2_shape.size();
        }

        Shape arg0_padded_shape = arg0_shape;
        Shape arg2_padded_shape = arg2_shape;
        // Trim trailing ones
        while (arg0_padded_shape.size() > 0 && arg0_padded_shape.back() == 1) {
            arg0_padded_shape.pop_back();
        }

        for (int64_t i = 0; (i < axis) && (arg0_padded_shape.size() < arg1_shape.size()); ++i) {
            arg0_padded_shape.insert(arg0_padded_shape.begin(), 1);
        }

        while (arg0_padded_shape.size() < arg1_shape.size()) {
            arg0_padded_shape.insert(arg0_padded_shape.end(), 1);
        }

        while (arg2_padded_shape.size() > 0 && arg2_padded_shape.back() == 1) {
            arg2_padded_shape.pop_back();
        }
        for (int64_t i = 0; (i < axis) && (arg2_padded_shape.size() < arg1_shape.size()); ++i) {
            arg2_padded_shape.insert(arg2_padded_shape.begin(), 1);
        }

        while (arg2_padded_shape.size() < arg1_shape.size()) {
            arg2_padded_shape.insert(arg2_padded_shape.end(), 1);
        }

        Shape arg0_squeezed_shape;
        AxisSet arg0_squeezed_axes;
        Shape arg2_squeezed_shape;
        AxisSet arg2_squeezed_axes;

        for (size_t i = 0, size = arg1_shape.size(); i < size; ++i) {
            if (arg0_padded_shape[i] == 1) {
                arg0_squeezed_axes.insert(i);
            } else {
                arg0_squeezed_shape.push_back(arg0_padded_shape[i]);
            }
            if (arg2_padded_shape[i] == 1) {
                arg2_squeezed_axes.insert(i);
            } else {
                arg2_squeezed_shape.push_back(arg2_padded_shape[i]);
            }
        }

        const CoordinateTransformBasic output_transform{arg1_shape};

        const auto arg0_strides = row_major_strides(arg0_squeezed_shape);
        const auto arg2_strides = row_major_strides(arg2_squeezed_shape);
        const auto output_strides = row_major_strides(arg1_shape);

        for (const Coordinate& output_coord : output_transform) {
            const auto arg0_coord = util::reduce(output_coord, arg0_squeezed_axes);
            const auto arg2_coord = util::reduce(output_coord, arg2_squeezed_axes);

            const size_t arg0_idx = coordinate_offset(arg0_coord, arg0_strides);
            const size_t arg1_idx = coordinate_offset(output_coord, output_strides);
            const size_t arg2_idx = coordinate_offset(arg2_coord, arg2_strides);
            const size_t output_idx = coordinate_offset(output_coord, output_strides);

            out[output_idx] = elementwise_functor(arg0[arg0_idx], arg1[arg1_idx], arg2[arg2_idx]);
        }
    }
    }
}
}  // namespace reference
}  // namespace ov
