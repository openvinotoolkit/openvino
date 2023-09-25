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

#include "ngraph/check.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/shape_util.hpp"
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

    if (shape_size(in_low_shape) == 1 && shape_size(in_high_shape) == 1 && shape_size(out_low_shape) == 1 &&
        shape_size(out_high_shape) == 1) {
        const size_t arg_size = shape_size(arg_shape);
        const auto q = [&](const T& a) {
            return quantize(a, *in_low, *in_high, *out_low, *out_high, levels_minus_one);
        };
        for (size_t i = 0; i < arg_size; ++i) {
            out[i] = q(arg[i]);
        }
        return;
    }

    std::vector<size_t> output_strides = compute_strides(arg_shape, arg_shape);
    std::vector<size_t> in_low_strides = compute_strides(arg_shape, in_low_shape);
    std::vector<size_t> in_high_strides = compute_strides(arg_shape, in_high_shape);
    std::vector<size_t> out_low_strides = compute_strides(arg_shape, out_low_shape);
    std::vector<size_t> out_high_strides = compute_strides(arg_shape, out_high_shape);

    size_t num_elements = shape_size(arg_shape);
    size_t input_inner_stride = num_elements;
    size_t in_low_inner_stride = 0;
    size_t in_high_inner_stride = 0;
    size_t out_low_inner_stride = 0;
    size_t out_high_inner_stride = 0;

    std::tie(in_low_inner_stride, input_inner_stride) =
        get_inner_stride(num_elements, arg_shape, in_low_shape, input_inner_stride);
    std::tie(in_high_inner_stride, input_inner_stride) =
        get_inner_stride(num_elements, arg_shape, in_high_shape, input_inner_stride);
    std::tie(out_low_inner_stride, input_inner_stride) =
        get_inner_stride(num_elements, arg_shape, out_low_shape, input_inner_stride);
    std::tie(out_high_inner_stride, input_inner_stride) =
        get_inner_stride(num_elements, arg_shape, out_high_shape, input_inner_stride);

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

    size_t in_low_stride = 0;
    size_t in_high_stride = 0;
    size_t out_low_stride = 0;
    size_t out_high_stride = 0;

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
        for (size_t i = 0; i < num_elements; i++) {
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
    size_t last = shape.back();
    auto it = std::find_if(shape.rbegin(), shape.rend(), [last](size_t dim) {
        return (last == 1 && dim > 1) || (last > 1 && dim == 1);
    });
    if (it == shape.rend()) {
        size_t num_elements = shape_size(shape);
        return std::tuple<size_t, size_t>{last == 1 ? 1 : num_elements,
                                          std::min(current_output_inner_stride, num_elements)};
    }
    size_t idx = std::distance(it, shape.rbegin()) + static_cast<int64_t>(shape.size());
    size_t inner_stride =
        std::accumulate(shape.begin() + idx, shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
    size_t weights_inner_stride = std::accumulate(output_shape.begin() + output_shape.size() - shape.size() + idx,
                                                  output_shape.end(),
                                                  static_cast<size_t>(1),
                                                  std::multiplies<size_t>());
    return std::tuple<size_t, size_t>{inner_stride, std::min(current_output_inner_stride, weights_inner_stride)};
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
        auto in_low_scalar = *in_low;
        auto in_high_scalar = *in_high;
        auto out_low_scalar = *out_low;
        auto out_high_scalar = *out_high;
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
        auto out_low_scalar = *out_low;
        auto out_high_scalar = *out_high;
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
        auto in_low_scalar = *in_low;
        auto in_high_scalar = *in_high;
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
