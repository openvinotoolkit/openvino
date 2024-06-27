// Copyright (C) 2018-2024 Intel Corporation
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
static inline T quantize(const T arg,
                         const T in_low,
                         const T in_high,
                         const T out_low,
                         const T out_high,
                         const T levels_minus_one) {
    if (arg <= std::min(in_low, in_high)) {
        return out_low;
    } else if (arg > std::max(in_low, in_high)) {
        return out_high;
    }
    return static_cast<T>(std::nearbyint((arg - in_low) / (in_high - in_low) * levels_minus_one) / levels_minus_one *
                              (out_high - out_low) +
                          out_low);
}

template <>
inline ov::bfloat16 quantize(const ov::bfloat16 arg,
                             const ov::bfloat16 in_low,
                             const ov::bfloat16 in_high,
                             const ov::bfloat16 out_low,
                             const ov::bfloat16 out_high,
                             const ov::bfloat16 levels_minus_one) {
    if (arg <= std::min(in_low, in_high)) {
        return out_low;
    } else if (arg > std::max(in_low, in_high)) {
        return out_high;
    }

    // make explicit convertion bf16->float to prevent implicit conversion bf16->float->bf16 on every operation
    const float arg_f = arg;
    const float in_low_f = in_low;
    const float in_high_f = in_high;
    const float out_low_f = out_low;
    const float out_high_f = out_high;
    const float levels_minus_one_f = levels_minus_one;

    return static_cast<ov::bfloat16>(std::nearbyint((arg_f - in_low_f) / (in_high_f - in_low_f) * levels_minus_one_f) /
                                         levels_minus_one_f * (out_high_f - out_low_f) +
                                     out_low_f);
}

static std::vector<size_t> compute_strides(const ov::Shape& out_shape, const ov::Shape& shape);

static std::tuple<size_t, size_t> get_inner_stride(size_t num_output_elements,
                                                   const ov::Shape& output_shape,
                                                   const ov::Shape& shape,
                                                   size_t current_output_inner_stride);

template <typename T, typename F>
static void fake_quantize_non_unit_inner_stride(const T* arg,
                                                const T* in_low,
                                                const T* in_high,
                                                const T* out_low,
                                                const T* out_high,
                                                T* out,
                                                const Shape& arg_shape,
                                                T levels_minus_one,
                                                size_t input_inner_stride,
                                                const F& get_outer_strides);

template <typename T, typename F>
static void fake_quantize_unit_inner_stride(const T* arg,
                                            const T* in_low,
                                            const T* in_high,
                                            const T* out_low,
                                            const T* out_high,
                                            T* out,
                                            const Shape& arg_shape,
                                            T levels_minus_one,
                                            size_t input_inner_stride,
                                            const F& get_outer_strides);

template <typename T, typename F>
static void fake_quantize_unit_output_intervals_inner_stride(const T* arg,
                                                             const T* in_low,
                                                             const T* in_high,
                                                             const T* out_low,
                                                             const T* out_high,
                                                             T* out,
                                                             const Shape& arg_shape,
                                                             T levels_minus_one,
                                                             size_t input_inner_stride,
                                                             const F& get_outer_strides);

template <typename T, typename F>
static void fake_quantize_unit_input_intervals_inner_stride(const T* arg,
                                                            const T* in_low,
                                                            const T* in_high,
                                                            const T* out_low,
                                                            const T* out_high,
                                                            T* out,
                                                            const Shape& arg_shape,
                                                            T levels_minus_one,
                                                            size_t input_inner_stride,
                                                            const F& get_outer_strides);

}  // namespace fake_quantize_details

template <typename T>
void fake_quantize(const T* arg,
                   const T* in_low,
                   const T* in_high,
                   const T* out_low,
                   const T* out_high,
                   T* out,
                   const Shape& arg_shape,
                   const Shape& in_low_shape,
                   const Shape& in_high_shape,
                   const Shape& out_low_shape,
                   const Shape& out_high_shape,
                   size_t levels,
                   const op::AutoBroadcastSpec& broadcast) {
    using namespace fake_quantize_details;

    T levels_minus_one = static_cast<T>(levels - 1);
    const size_t arg_size = shape_size(arg_shape);

    if (shape_size(in_low_shape) == 1 && shape_size(in_high_shape) == 1 && shape_size(out_low_shape) == 1 &&
        shape_size(out_high_shape) == 1) {
        for (size_t i = 0; i < arg_size; ++i) {
            out[i] = quantize(arg[i], *in_low, *in_high, *out_low, *out_high, levels_minus_one);
        }
        return;
    }

    // clang-format off
    /*
     * ---------------------------------------------------
     * Overview:
     *  Numpy broadcasted input tensors can be partitioned into two: outer and inner part (which also defines inner
     *  stride as a product of inner part), so N-dimensional tensors can be processed using two loops.
     *
     *  For example with two inputs [2, 2, 3, 4] and [1, 1, 3, 4] we can have:
     *      input 1 with shape [2, 2, 3, 4] can be divided into outer part [2, 2] and inner part [3, 4]
     *      with inner stride = 12 (3 * 4).
     *      input 2 with shape [1, 1, 3, 4] can be divided into outer part [1, 1]
     *      and inner part [3, 4] with inner stride = 12 (3 * 4)
     *
     *      Having that, those inputs can be processed by the following:
     *
     *      output_shape = {2, 2, 3, 4};
     *      output_inner_stride = 12;
     *      for (i = 0; i < shape_size(shape); i += output_inner_stride) {
     *          first_input_stride = i;
     *          second_input_stride = 0;
     *          for (j = 0; j < 12; j++) {
     *              *out++ = f(first_input[first_input_stride + j], second_input[second_input_stride + j]);
     *          }
     *      }
     *
     * ---------------------------------------------------
     * How the partitioning is done:
     *  Partitioning process starts with the last dimension of input tensor shape and it stops when either one of below
     *  occurs:
     *      - if the last dimension is equal to 1, partitioning stops at the dimension that is greater than 1 (this
     *        dimension is not included in the inner part),
     *      - if the last dimension is greater than 1, partitioning stops at the dimension that is equal to 1 (this
     *        dimension is not included in the inner part).
     *
     *  Examples:
     *      tensor_shape=[2, 3, 4, 5], inner_part = [2, 3, 4, 5], inner_stride = 120
     *      tensor_shape=[1, 1, 4, 5], inner_part = [4, 5], inner_stride = 20
     *      tensor_shape=[2, 3, 1, 1], inner_part = [1, 1], inner_stride = 1
     *
     *
     * ---------------------------------------------------
     * How the output inner stride is calculated:
     *  Inner part (and inner stride) for every input tensor is determined. Then the size of output inner part is the
     *  size of inner part with the fewest number of dimensions.
     *
     *  Example with 5 inputs:
     *      input 1 shape [2, 3, 4, 5], inner_part = [2, 3, 4, 5], inner_stride = 120
     *      input 2 shape [1, 3, 4, 5], inner_part = [3, 4, 5], inner_stride = 60
     *      input 3 shape [2, 3, 1, 1], inner_part = [1, 1], inner_stride = 1
     *      input 4 shape [2, 1, 1, 1], inner_part = [1, 1, 1], inner_stride = 1
     *      input 5 shape [1, 1, 1, 1], inner_part = [1, 1, 1, 1], inner_stride = 1
     *
     *      output shape [2, 3, 4, 5], inner_part = [4, 5], inner_stride = 20
     *
     *      Inner part with fewest number of elements is [1, 1] for input 3. So the inner part for output shape is [4, 5]
     *      and output inner stride is 20.
     */
    // clang-format on

    std::vector<size_t> output_strides = compute_strides(arg_shape, arg_shape);
    std::vector<size_t> in_low_strides = compute_strides(arg_shape, in_low_shape);
    std::vector<size_t> in_high_strides = compute_strides(arg_shape, in_high_shape);
    std::vector<size_t> out_low_strides = compute_strides(arg_shape, out_low_shape);
    std::vector<size_t> out_high_strides = compute_strides(arg_shape, out_high_shape);

    size_t input_inner_stride = arg_size;
    size_t in_low_inner_stride = 0;
    size_t in_high_inner_stride = 0;
    size_t out_low_inner_stride = 0;
    size_t out_high_inner_stride = 0;

    std::tie(in_low_inner_stride, input_inner_stride) =
        get_inner_stride(arg_size, arg_shape, in_low_shape, input_inner_stride);
    std::tie(in_high_inner_stride, input_inner_stride) =
        get_inner_stride(arg_size, arg_shape, in_high_shape, input_inner_stride);
    std::tie(out_low_inner_stride, input_inner_stride) =
        get_inner_stride(arg_size, arg_shape, out_low_shape, input_inner_stride);
    std::tie(out_high_inner_stride, input_inner_stride) =
        get_inner_stride(arg_size, arg_shape, out_high_shape, input_inner_stride);

    auto get_outer_strides =
        [&output_strides, &in_low_strides, &in_high_strides, &out_low_strides, &out_high_strides](size_t flat_index) {
            size_t in_low_stride = 0;
            size_t in_high_stride = 0;
            size_t out_low_stride = 0;
            size_t out_high_stride = 0;

            for (size_t i = 0; i < output_strides.size(); i++) {
                size_t div = flat_index / output_strides[i];
                flat_index = flat_index % output_strides[i];
                in_low_stride += div * in_low_strides[i];
                in_high_stride += div * in_high_strides[i];
                out_low_stride += div * out_low_strides[i];
                out_high_stride += div * out_high_strides[i];
            }

            return std::tuple<size_t, size_t, size_t, size_t>{in_low_stride,
                                                              in_high_stride,
                                                              out_low_stride,
                                                              out_high_stride};
        };

    if (in_low_inner_stride > 1 && in_high_inner_stride > 1 && out_low_inner_stride > 1 && out_high_inner_stride > 1) {
        fake_quantize_non_unit_inner_stride(arg,
                                            in_low,
                                            in_high,
                                            out_low,
                                            out_high,
                                            out,
                                            arg_shape,
                                            levels_minus_one,
                                            input_inner_stride,
                                            get_outer_strides);
    } else if (in_low_inner_stride == 1 && in_high_inner_stride == 1 && out_low_inner_stride == 1 &&
               out_high_inner_stride == 1) {
        fake_quantize_unit_inner_stride(arg,
                                        in_low,
                                        in_high,
                                        out_low,
                                        out_high,
                                        out,
                                        arg_shape,
                                        levels_minus_one,
                                        input_inner_stride,
                                        get_outer_strides);

    } else if (in_low_inner_stride > 1 && in_high_inner_stride > 1 && out_low_inner_stride == 1 &&
               out_high_inner_stride == 1) {
        fake_quantize_unit_output_intervals_inner_stride(arg,
                                                         in_low,
                                                         in_high,
                                                         out_low,
                                                         out_high,
                                                         out,
                                                         arg_shape,
                                                         levels_minus_one,
                                                         input_inner_stride,
                                                         get_outer_strides);

    } else if (in_low_inner_stride == 1 && in_high_inner_stride == 1 && out_low_inner_stride > 1 &&
               out_high_inner_stride > 1) {
        fake_quantize_unit_input_intervals_inner_stride(arg,
                                                        in_low,
                                                        in_high,
                                                        out_low,
                                                        out_high,
                                                        out,
                                                        arg_shape,
                                                        levels_minus_one,
                                                        input_inner_stride,
                                                        get_outer_strides);
    } else {
        size_t in_low_stride = 0;
        size_t in_high_stride = 0;
        size_t out_low_stride = 0;
        size_t out_high_stride = 0;

        for (size_t i = 0; i < arg_size; i++) {
            std::tie(in_low_stride, in_high_stride, out_low_stride, out_high_stride) = get_outer_strides(i);
            *out++ = quantize(*arg++,
                              *(in_low + in_low_stride),
                              *(in_high + in_high_stride),
                              *(out_low + out_low_stride),
                              *(out_high + out_low_stride),
                              levels_minus_one);
        }
    }
}

namespace fake_quantize_details {
std::vector<size_t> compute_strides(const ov::Shape& out_shape, const ov::Shape& shape) {
    size_t stride = 1;
    size_t out_rank = out_shape.size();
    size_t shape_rank = shape.size();
    std::vector<size_t> strides(out_rank);
    for (size_t i = 0; i < out_rank; i++) {
        if (i < shape_rank && shape[shape_rank - i - 1] == out_shape[out_rank - i - 1]) {
            strides[out_rank - i - 1] = stride;
            stride *= shape[shape_rank - i - 1];
        } else {
            strides[out_rank - i - 1] = 0;
        }
    }
    return strides;
}

std::tuple<size_t, size_t> get_inner_stride(size_t num_output_elements,
                                            const ov::Shape& output_shape,
                                            const ov::Shape& shape,
                                            size_t current_output_inner_stride) {
    if (shape.size() == 0)
        return std::tuple<size_t, size_t>{1, std::min(current_output_inner_stride, num_output_elements)};
    const size_t last = shape.back();
    auto it = std::find_if(shape.rbegin(), shape.rend(), [last](size_t dim) {
        return (last == 1 && dim > 1) || (last > 1 && dim == 1);
    });
    if (it == shape.rend()) {
        const auto num_elements = shape_size(shape);
        return {num_elements,
                last == 1 ? current_output_inner_stride : std::min(current_output_inner_stride, num_elements)};
    }
    const auto idx = std::distance(it, shape.rbegin()) + static_cast<std::ptrdiff_t>(shape.size());
    const auto inner_stride = shape_size(shape.begin() + idx, shape.end());
    const auto output_inner_stride =
        shape_size(output_shape.begin() + (output_shape.size() - shape.size() + idx), output_shape.end());
    return {inner_stride, std::min(current_output_inner_stride, output_inner_stride)};
}

template <typename T, typename F>
static void transform(const T* first1, const T* const last1, const T* first2, const T* first3, T* out, const F& f) {
    while (first1 < last1) {
        *out++ = f(*first1++, *first2++, *first3++);
    }
}

template <typename T, typename F>
static void transform(const T* first1,
                      const T* const last1,
                      const T* first2,
                      const T* first3,
                      const T* first4,
                      const T* first5,
                      T* out,
                      const F& f) {
    while (first1 < last1) {
        *out++ = f(*first1++, *first2++, *first3++, *first4++, *first5++);
    }
}

template <typename T, typename F1, typename F2>
static void fake_quantize_loop(const Shape& arg_shape,
                               const T* arg,
                               const T* in_low,
                               const T* in_high,
                               const T* out_low,
                               const T* out_high,
                               T* out,
                               size_t input_inner_stride,
                               const F1& get_outer_strides,
                               const F2& quantize_loop) {
    size_t in_low_stride = 0;
    size_t in_high_stride = 0;
    size_t out_low_stride = 0;
    size_t out_high_stride = 0;

    for (size_t i = 0; i < shape_size(arg_shape); i += input_inner_stride) {
        std::tie(in_low_stride, in_high_stride, out_low_stride, out_high_stride) = get_outer_strides(i);
        quantize_loop(arg,
                      arg + input_inner_stride,
                      in_low + in_low_stride,
                      in_high + in_high_stride,
                      out_low + out_low_stride,
                      out_high + out_high_stride,
                      out);
        arg += input_inner_stride;
        out += input_inner_stride;
    }
}

template <typename T, typename F>
void fake_quantize_non_unit_inner_stride(const T* arg,
                                         const T* in_low,
                                         const T* in_high,
                                         const T* out_low,
                                         const T* out_high,
                                         T* out,
                                         const Shape& arg_shape,
                                         T levels_minus_one,
                                         size_t input_inner_stride,
                                         const F& get_outer_strides) {
    fake_quantize_loop(arg_shape,
                       arg,
                       in_low,
                       in_high,
                       out_low,
                       out_high,
                       out,
                       input_inner_stride,
                       get_outer_strides,
                       [levels_minus_one](const T* input,
                                          const T* const input_end,
                                          const T* in_low,
                                          const T* in_high,
                                          const T* out_low,
                                          const T* out_high,
                                          T* out) {
                           transform(input,
                                     input_end,
                                     in_low,
                                     in_high,
                                     out_low,
                                     out_high,
                                     out,
                                     [levels_minus_one](T input, T in_low, T in_high, T out_low, T out_high) {
                                         return quantize(input, in_low, in_high, out_low, out_high, levels_minus_one);
                                     });
                       });
}

template <typename T, typename F>
void fake_quantize_unit_inner_stride(const T* arg,
                                     const T* in_low,
                                     const T* in_high,
                                     const T* out_low,
                                     const T* out_high,
                                     T* out,
                                     const Shape& arg_shape,
                                     T levels_minus_one,
                                     size_t input_inner_stride,
                                     const F& get_outer_strides) {
    auto quantize_with_scalar_intervals = [levels_minus_one](const T* input,
                                                             const T* const input_end,
                                                             const T* in_low,
                                                             const T* in_high,
                                                             const T* out_low,
                                                             const T* out_high,
                                                             T* out) {
        const auto in_low_scalar = *in_low;
        const auto in_high_scalar = *in_high;
        const auto out_low_scalar = *out_low;
        const auto out_high_scalar = *out_high;
        std::transform(input,
                       input_end,
                       out,
                       [levels_minus_one, in_low_scalar, in_high_scalar, out_low_scalar, out_high_scalar](T input) {
                           return quantize(input,
                                           in_low_scalar,
                                           in_high_scalar,
                                           out_low_scalar,
                                           out_high_scalar,
                                           levels_minus_one);
                       });
    };

    fake_quantize_loop(arg_shape,
                       arg,
                       in_low,
                       in_high,
                       out_low,
                       out_high,
                       out,
                       input_inner_stride,
                       get_outer_strides,
                       quantize_with_scalar_intervals);
}

template <typename T, typename F>
void fake_quantize_unit_output_intervals_inner_stride(const T* arg,
                                                      const T* in_low,
                                                      const T* in_high,
                                                      const T* out_low,
                                                      const T* out_high,
                                                      T* out,
                                                      const Shape& arg_shape,
                                                      T levels_minus_one,
                                                      size_t input_inner_stride,
                                                      const F& get_outer_strides) {
    auto quantize_with_scalar_output_intervals = [levels_minus_one](const T* input,
                                                                    const T* const input_end,
                                                                    const T* in_low,
                                                                    const T* in_high,
                                                                    const T* out_low,
                                                                    const T* out_high,
                                                                    T* out) {
        const auto out_low_scalar = *out_low;
        const auto out_high_scalar = *out_high;
        transform(input,
                  input_end,
                  in_low,
                  in_high,
                  out,
                  [levels_minus_one, out_low_scalar, out_high_scalar](T input, T in_low, T in_high) {
                      return quantize(input, in_low, in_high, out_low_scalar, out_high_scalar, levels_minus_one);
                  });
    };

    fake_quantize_loop(arg_shape,
                       arg,
                       in_low,
                       in_high,
                       out_low,
                       out_high,
                       out,
                       input_inner_stride,
                       get_outer_strides,
                       quantize_with_scalar_output_intervals);
}

template <typename T, typename F>
void fake_quantize_unit_input_intervals_inner_stride(const T* arg,
                                                     const T* in_low,
                                                     const T* in_high,
                                                     const T* out_low,
                                                     const T* out_high,
                                                     T* out,
                                                     const Shape& arg_shape,
                                                     T levels_minus_one,
                                                     size_t input_inner_stride,
                                                     const F& get_outer_strides) {
    auto quantize_with_scalar_input_intervals = [levels_minus_one](const T* input,
                                                                   const T* const input_end,
                                                                   const T* in_low,
                                                                   const T* in_high,
                                                                   const T* out_low,
                                                                   const T* out_high,
                                                                   T* out) {
        const auto in_low_scalar = *in_low;
        const auto in_high_scalar = *in_high;
        transform(input,
                  input_end,
                  out_low,
                  out_high,
                  out,
                  [levels_minus_one, in_low_scalar, in_high_scalar](T input, T out_low, T out_high) {
                      return quantize(input, in_low_scalar, in_high_scalar, out_low, out_high, levels_minus_one);
                  });
    };

    fake_quantize_loop(arg_shape,
                       arg,
                       in_low,
                       in_high,
                       out_low,
                       out_high,
                       out,
                       input_inner_stride,
                       get_outer_strides,
                       quantize_with_scalar_input_intervals);
}

}  // namespace fake_quantize_details

}  // namespace reference
}  // namespace ov
