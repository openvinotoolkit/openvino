// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <vector>

#include "openvino/reference/autobroadcast_binop.hpp"
#include "openvino/reference/convert.hpp"

namespace ov {
namespace reference {
namespace func {
/**
 * @brief Emulation of conversion fp16 value to f8e5m2 format
 *
 * @param arg_f       Pointer to the input data.
 * @param out_f       Pointer to the otuput data.
 * @param count       Number of elements in the data input.
 * @param use_clamp   If use clamp.
 */
void emulate_f8e5m2_on_fp16(const float16* const arg_f, float16* out_f, size_t count, bool use_clamp = true);

/**
 * @brief Emulation of conversion fp16 value to f8e4m3 format
 *
 * @param arg_f       Pointer to the input data.
 * @param out_f       Pointer to the otuput data.
 * @param count       Number of elements in the data input.
 * @param use_clamp   If use clamp.
 *
 * Exponent denormal values 0 -7
 * Exponent normal values 1..15 -6..8 (7 - exponent)
 * Exponent NaN values 15 8
 *
 */
void emulate_f8e4m3_on_fp16(const float16* arg_f, float16* out_f, size_t count, bool use_clamp = true);
}  // namespace func

namespace fake_convert_details {
template <bool INVERT, typename T>
inline T scale_shift(T val, T scale, T shift) {
    return INVERT ? static_cast<T>((val + shift) / scale) : static_cast<T>(val * scale - shift);
}
/**
 * @brief Apply scale and shift for the data input.
 * The scale_shape and shift_shape should be equal and numpy-broadcastable to the data_shape.
 *
 * @param data                   Pointer to the input data.
 * @param scale                  Pointer to the input scale.
 * @param shift                  Pointer to the input shift.
 * @param data_shape             Shape of the input data.
 * @param scale_shape            Shape of the input scale.
 * @param shift_shape            Shape of the input shift.
 * @param invert                 Flag denoting applying scale before (if false) or after conversion (if true).
 */
template <typename T>
void apply_scale_shift(T* out,
                       const T* data,
                       const T* scale,
                       const T* shift,
                       const Shape& data_shape,
                       const Shape& scale_shape,
                       const Shape& shift_shape,
                       const bool invert = false) {
    const auto scale_shift_func = invert ? scale_shift<true, T> : scale_shift<false, T>;
    const auto scale_size = shape_size(scale_shape);
    const auto data_size = shape_size(data_shape);

    if (scale_size == 1) {  // per tensor scale, probably for activation
        T scale_val = scale[0];
        T shift_val = shift[0];
        for (size_t j = 0; j < data_size; ++j) {
            out[j] = scale_shift_func(data[j], scale_val, shift_val);
        }
        return;
    }

    if (scale_shape.size() > 1 && scale_shape[0] == 1 && data_shape.size() == scale_shape.size() &&
        data_shape[1] == scale_shape[1] && scale_size == data_shape[1]) {  // per channel scale for DW activations
        const auto step = shape_size(data_shape.begin() + 2, data_shape.end());
        for (size_t bs = 0; bs < data_shape[0]; ++bs) {
            for (size_t i = 0; i < scale_size; i++) {
                T scale_val = scale[i];
                T shift_val = shift[i];
                for (size_t j = 0; j < step; ++j) {
                    out[j] = scale_shift_func(data[j], scale_val, shift_val);
                }
                data += step;
                out += step;
            }
        }
        return;
    }

    // The specific cases above are optimized verions of the broadcast for specific case
    // Autobroadcast helper is generic approach for broadcast
    autobroadcast_select(data,
                         scale,
                         shift,
                         out,
                         data_shape,
                         scale_shape,
                         shift_shape,
                         op::AutoBroadcastType::NUMPY,
                         scale_shift_func);
}
/**
 * @brief Call conversion of fp16 value to the desired destination type
 *
 * @param arg                  Pointer to the input data.
 * @param out                  Pointer to the otuput data.
 * @param count                Number of elements in the data input.
 * @param destination_type     Name of the destination type.
 */
void apply_conversion(const float16* data, float16* out, size_t element_count, const element::Type& destination_type);
}  // namespace fake_convert_details

/**
 * @brief Reference implementation of the FakeConvert operation specialized for float16 type.
 * It emulates format specified by "destination_type" on the input type.
 * FakeConvert performs element-wise quantization of floating-point input values into a set of values corresponding to a
 * target low-precision floating-point type.
 *
 *
 * @param data                   Pointer to the input data.
 * @param scale                  Pointer to the input scale.
 * @param shift                  Pointer to the input shift.
 * @param out                    Pointer to the output data.
 * @param data_shape             Shape of the input data.
 * @param scale_shape            Shape of the input scale.
 * @param shift_shape            Shape of the input shift.
 * @param destination_type       Name of the destination type.
 */
template <typename T, typename std::enable_if<std::is_same<T, float16>::value, bool>::type = true>
void fake_convert(const T* data,
                  const T* scale,
                  const T* shift,
                  T* out,
                  const Shape& data_shape,
                  const Shape& scale_shape,
                  const Shape& shift_shape,
                  const element::Type& destination_type) {
    const size_t element_count = shape_size(data_shape);
    fake_convert_details::apply_scale_shift<float16>(out,
                                                     data,
                                                     scale,
                                                     shift,
                                                     data_shape,
                                                     scale_shape,
                                                     shift_shape,
                                                     false);
    fake_convert_details::apply_conversion(out, out, element_count, destination_type);
    fake_convert_details::apply_scale_shift<float16>(out,
                                                     out,
                                                     scale,
                                                     shift,
                                                     data_shape,
                                                     scale_shape,
                                                     shift_shape,
                                                     true);
}

/**
 * @brief Reference implementation of the FakeConvert operation for floating-point types, with middle conversion
 * to float16. It emulates format specified by "destination_type" on the input type. FakeConvert performs
 * element-wise quantization of floating-point input values into a set of values corresponding to a target low-precision
 * floating-point type.
 *
 *
 * @param data                   Pointer to the input data.
 * @param scale                  Pointer to the input scale.
 * @param shift                  Pointer to the input shift.
 * @param out                    Pointer to the output data.
 * @param data_shape             Shape of the input data.
 * @param scale_shape            Shape of the input scale.
 * @param shift_shape            Shape of the input shift.
 * @param destination_type       Name of the destination type.
 */
template <typename T, typename std::enable_if<!std::is_same<T, float16>::value, bool>::type = true>
void fake_convert(const T* data,
                  const T* scale,
                  const T* shift,
                  T* out,
                  const Shape& data_shape,
                  const Shape& scale_shape,
                  const Shape& shift_shape,
                  const element::Type& destination_type) {
    const size_t element_count = shape_size(data_shape);
    fake_convert_details::apply_scale_shift<T>(out, data, scale, shift, data_shape, scale_shape, shift_shape, false);

    std::vector<ov::float16> tmp_fp16(element_count, 0.f);
    reference::convert(out, tmp_fp16.data(), element_count);
    fake_convert_details::apply_conversion(tmp_fp16.data(), tmp_fp16.data(), element_count, destination_type);
    reference::convert(tmp_fp16.data(), out, element_count);

    fake_convert_details::apply_scale_shift<T>(out, out, scale, shift, data_shape, scale_shape, shift_shape, true);
}

/**
 * @brief Reference implementation of the FakeConvert operation, with default `shift` input.
 */
template <typename T>
void fake_convert(const T* data,
                  const T* scale,
                  T* out,
                  const Shape& data_shape,
                  const Shape& scale_shape,
                  const element::Type& destination_type) {
    const auto shift = std::vector<T>(shape_size(scale_shape), 0.f);
    fake_convert<T>(data, scale, shift.data(), out, data_shape, scale_shape, scale_shape, destination_type);
}

}  // namespace reference
}  // namespace ov
