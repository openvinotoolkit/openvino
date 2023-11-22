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

#include "openvino/reference/autobroadcast_binop.hpp"

namespace ov {
namespace reference {
namespace func {

/**
 * @brief Emulation of conversion fp16 value to f8e5m2 format
 *
 * @param arg       Pointer to the input data.
 * @param out       Pointer to the otuput data.
 * @param count     Number of elements in the data input.
 */
static void emulate_f8e5m2_on_fp16(const float16* const arg_f, float16* out_f, size_t count) {
    const auto arg_u = reinterpret_cast<const uint16_t*>(arg_f);
    auto out_u = reinterpret_cast<uint16_t*>(out_f);
    uint16_t val_bit_repr;

    constexpr auto exp_bits = 5;
    constexpr auto mbits = 8;
    constexpr auto non_mant_bits = exp_bits + 1;  /// exponent + sign
    constexpr auto lshift = 10 - (mbits - non_mant_bits);
    constexpr uint16_t mask_mant = static_cast<uint16_t>(0xFFFF << lshift);  /// 1111111111111111 -> 1 11111 1100000000
    constexpr uint16_t grs_bitmask = 0x00FF;  /// 0 00000 0011111111, grs denotes guard, round, sticky bits
    constexpr uint16_t rne_tie = 0x0180;      /// 0 00000 0110000000, rne denotes round to nearest even
    constexpr uint16_t fp16_inf = 0x7C00;

    for (size_t i = 0; i < count; ++i) {
        /// converts float number to half precision in round-to-nearest-even mode and returns half with converted value.
        val_bit_repr = arg_u[i];
        /// 0x7c00 = 0111110000000000 - exponent mask
        /// s 11111 xxx xxxx xxxx - is nan (if some x is 1) or inf (if all x is 0)
        /// 0x7800 is 0111100000000000 and 0x400 is 0000010000000000
        /// number is not normal if all exponent is 1 or 0
        /// 0x7f00 is 0 11111 1100000000
        /// 0x7b00 is 0 11110 1100000000
        const bool can_round = ((val_bit_repr & 0x7F00) < 0x7B00) ? true : false;
        /// s 11111 xxx xxxx xxxx - is nan (if some x is 1) or inf (if all x is 0)
        const bool is_naninf = ((val_bit_repr & fp16_inf) == fp16_inf) ? true : false;
        /* nearest rounding masks */
        /// grs_bitmask - grs_bitmask is 0 00000 0011111111 or 0 00000 00grs11111
        uint16_t rnmask = (val_bit_repr & grs_bitmask);
        /// rne_tie - 0x180 is      0 00000 0110000000 or 384.0
        uint16_t rnmask_tie = (val_bit_repr & rne_tie);

        if (!is_naninf && can_round) {
            /* round to nearest even, if rne_mask is enabled */
            /* 0 00000 0010000000, find grs patterns */
            // 0xx - do nothing
            // 100 - this is a tie : round up if the mantissa's bit just before G is 1, else do nothing
            // 101, 110, 111 - round up > 0x0080
            val_bit_repr += (((rnmask > 0x0080) || (rnmask_tie == rne_tie)) << lshift);
        }
        val_bit_repr &= mask_mant; /* truncation */
        out_u[i] = val_bit_repr;
    }
}

/**
 * @brief Emulation of conversion fp16 value to f8e4m3 format
 *
 * @param arg       Pointer to the input data.
 * @param out       Pointer to the otuput data.
 * @param count     Number of elements in the data input.
 *
 * Exponent denormal values 0 -7
 * Exponent normal values 1..15 -6..8 (7 - exponent)
 * Exponent NaN values 15 8
 *
 */
static void emulate_f8e4m3_on_fp16(const float16* arg_f, float16* out_f, size_t count) {
    const auto arg_u = reinterpret_cast<const uint16_t*>(arg_f);
    auto out_u = reinterpret_cast<uint16_t*>(out_f);
    uint16_t val_bit_repr;

    constexpr auto use_clamp = true;
    constexpr auto exp_bits = 5;
    constexpr auto mbits = 9;
    constexpr auto non_mant_bits = exp_bits + 1;  /// exponent + sign
    constexpr auto lshift = 10 - (mbits - non_mant_bits);
    constexpr auto fp16_exp_bias = 15;
    constexpr auto f8e4m3_min_val = 0.001953125f;  /// 2**-9

    constexpr uint16_t mask_mant = static_cast<uint16_t>(0xFFFF << lshift);  /// 1111111111111111 -> 1 11111 1111000000
    constexpr uint16_t grs_bitmask = 0x007F;  /// 0 00000 0001111111, grs denotes guard, round, sticky bits
    constexpr uint16_t rne_tie = 0x00C0;      /// 0 00000 0011000000, rne denotes round to nearest even
    constexpr uint16_t rne_mask = 1;
    constexpr uint16_t fp16_inf = 0x7C00;

    for (size_t i = 0; i < count; ++i) {
        val_bit_repr = arg_u[i];
        /* flush values below 1-4-3 (offset=4) subnormal range to zero */
        if (std::abs(static_cast<float>(arg_f[i])) < f8e4m3_min_val) {
            val_bit_repr = 0;
        }

        short exp_h = static_cast<short>((val_bit_repr & fp16_inf) >> 10) -
                      fp16_exp_bias;  /// 0111110000000000 -> 0000000000011111 - 15, biased exponent for fp16
        const short sign_h = (val_bit_repr & 0x8000);  /// & 1 00000 0000000000
        short mantissa_h = (val_bit_repr & 0x03FF);    /// & 0 00000 1111111111
        ///(val_bit_repr && 0111111111111111) < 0 10010 1110000000 (19326)
        const bool can_round = ((val_bit_repr & 0x7FFF) < 0b101111110000000) ? true : false;
        bool is_naninf = ((val_bit_repr & fp16_inf) == fp16_inf) ? true : false;

        int dshift = 0;
        if (exp_h > 8) {  // too large, set it to NaN or inf
            is_naninf = true;
            if (use_clamp) {
                exp_h = 8;
                mantissa_h = 0b0000001100000000;
            } else {
                mantissa_h = 0;
                exp_h = 16;
            }
        } else if (exp_h < -9) {  /// -13, -12, -11 for rounding
            /* flush values below 1-4-3 (offset=4) subnormal range to zero */
            exp_h = -15;
            mantissa_h = 0;
        }

        if (exp_h == 8 && mantissa_h >= 0b0000001100000000) {
            mantissa_h = 0b0000001100000000;
        }
        /* nearest rounding masks, & 0 00000 000 111 1111 - mantissa bits below f8e4m3 (grs) */
        const uint16_t rnmask = (mantissa_h & grs_bitmask);
        /* & 0 00000 0011000000 - edge between f8e4m3 and fp16 mantissa */
        const uint16_t rnmask_tie = (mantissa_h & rne_tie);
        if (!is_naninf && can_round && rne_mask) {
            /* round to nearest even, if rne_mask is enabled */
            /// rnmask > 0 00000 0001000000(64) or 0 00000 0011000000 - edge bits is 1
            /// += 0 00000 0010000000
            mantissa_h += (((rnmask > 0x0040) || (rnmask_tie == rne_tie)) << lshift);
        }
        if (exp_h < -6) { /* handle denormals -11, -10, -9, dshift 1, 2, 3 */
            dshift = (-6 - exp_h);
            mantissa_h = mantissa_h >> dshift;
        }
        mantissa_h &= mask_mant; /* truncation */
        mantissa_h <<= dshift;
        mantissa_h += ((exp_h + 15) << 10);
        val_bit_repr = mantissa_h | sign_h;
        out_u[i] = val_bit_repr;
    }
}

template <typename T>
void apply_scale_shift(T* out,
                       const T* data,
                       const T* scale,
                       const T* shift,
                       const Shape& data_shape,
                       const Shape& scale_shape,
                       const Shape& shift_shape,
                       const bool invert = false) {
    const auto scale_size = shape_size(scale_shape);
    const auto data_size = shape_size(data_shape);

    if (scale_size == 1) {  // per tensor scale, probably for activation
        T s = scale[0];
        T o = shift[0];
        if (invert) {
            for (size_t j = 0; j < data_size; ++j) {
                out[j] = (data[j] + o) / s;
            }
        } else {
            for (size_t j = 0; j < data_size; ++j) {
                out[j] = data[j] * s - o;  // o = quntized(o * s)
            }
        }
        return;
    }

    if (scale_shape.size() > 1 && scale_shape[0] == 1 && data_shape.size() == scale_shape.size() &&
        data_shape[1] == scale_shape[1] && scale_size == data_shape[1]) {  // per channel scale for DW activations
        size_t step = 1;
        for (size_t i = 2; i < data_shape.size(); i++) {
            step *= data_shape[i];
        }

        for (size_t bs = 0; bs < data_shape[0]; ++bs) {
            for (size_t i = 0; i < scale_size; i++) {
                T s = scale[i];
                T o = shift[i];
                if (invert) {
                    for (size_t j = 0; j < step; ++j) {
                        out[j] = (data[j] + o) / s;
                    }
                } else {
                    for (size_t j = 0; j < step; ++j) {
                        out[j] = data[j] * s - o;  // o = quntized(o * s)
                    }
                }
                data += step;
                out += step;
            }
        }
        return;
    }

    auto scale_shift_func = invert ? [](T elem, T s, T o) -> T {
        return static_cast<T>((elem + o) / s);
    }
    : [](T elem, T s, T o) -> T {
          return static_cast<T>(elem * s - o);
      };

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
}  // namespace func

static void apply_conversion(const float16* data,
                             float16* out,
                             size_t element_count,
                             const std::string& destination_type) {
    if (destination_type == "f8e5m2") {
        reference::func::emulate_f8e5m2_on_fp16(data, out, element_count);
    } else if (destination_type == "f8e4m3") {
        reference::func::emulate_f8e4m3_on_fp16(data, out, element_count);
    } else {
        OPENVINO_THROW("Unsupported destination type.");
    }
}

template <typename T, typename std::enable_if<std::is_same<T, float16>::value, bool>::type = true>
void fake_convert(const T* data,
                  const T* scale,
                  const T* shift,
                  T* out,
                  const Shape& data_shape,
                  const Shape& scale_shape,
                  const Shape& shift_shape,
                  const std::string& destination_type) {
    const size_t element_count = shape_size(data_shape);
    reference::func::apply_scale_shift<float16>(out,
                                                                data,
                                                                scale,
                                                                shift,
                                                                data_shape,
                                                                scale_shape,
                                                                shift_shape,
                                                                false);
    apply_conversion(out, out, element_count, destination_type);
    reference::func::apply_scale_shift<float16>(out,
                                                                out,
                                                                scale,
                                                                shift,
                                                                data_shape,
                                                                scale_shape,
                                                                shift_shape,
                                                                true);
}

template <typename T, typename std::enable_if<!std::is_same<T, float16>::value, bool>::type = true>
void fake_convert(const T* data,
                  const T* scale,
                  const T* shift,
                  T* out,
                  const Shape& data_shape,
                  const Shape& scale_shape,
                  const Shape& shift_shape,
                  const std::string& destination_type) {
    const size_t element_count = shape_size(data_shape);
    reference::func::apply_scale_shift<T>(out,
                                                          data,
                                                          scale,
                                                          shift,
                                                          data_shape,
                                                          scale_shape,
                                                          shift_shape,
                                                          false);

    std::vector<ov::float16> tmp_fp16;
    tmp_fp16.reserve(element_count);
    reference::convert(out, tmp_fp16.data(), element_count);
    apply_conversion(tmp_fp16.data(), tmp_fp16.data(), element_count, destination_type);
    reference::convert(tmp_fp16.data(), out, element_count);

    reference::func::apply_scale_shift<T>(out,
                                                          out,
                                                          scale,
                                                          shift,
                                                          data_shape,
                                                          scale_shape,
                                                          shift_shape,
                                                          true);
}

template <typename T>
void fake_convert(const T* data,
                  const T* scale,
                  T* out,
                  const Shape& data_shape,
                  const Shape& scale_shape,
                  const std::string& destination_type) {
    const auto shift = std::vector<T>(shape_size(scale_shape), 0.f);
    fake_convert<T>(data, scale, shift.data(), out, data_shape, scale_shape, scale_shape, destination_type);
}

}  // namespace reference
}  // namespace ov
