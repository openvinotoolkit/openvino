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
namespace fake_convert_details {

/**
 * @brief Emulation of conversion fp16 value to f8e5m2 format
 *
 * @param arg       Pointer to the input data.
 * @param out       Pointer to the otuput data.
 * @param count     Number of elements in the data input.
 */
template <typename T>
void emulate_f8e5m2_on_fp16(const T* const arg, T* out, size_t count) {
    typedef union half_t {
        unsigned short u;
        T f;
    } __half_t;

    constexpr auto exp_bits = 5;
    constexpr auto mbits = 8;
    constexpr auto non_mant_bits = exp_bits + 1;  // exponent + sign
    constexpr auto lshift = 10 - (mbits - non_mant_bits);
    constexpr unsigned short mask_mant =
        static_cast<unsigned short>(0xFFFF << lshift);  // 1111111111111111 -> 1 11111 1100000000
    constexpr unsigned short grs_bitmask = 0x00FF;      // 0 00000 0011111111 - guard, round, sticky bits
    constexpr unsigned short rne_tie = 0x0180;          // 0 00000 0110000000
    constexpr unsigned short fp16_inf = 0x7C00;

    __half_t h;
    for (size_t i = 0; i < count; ++i) {
        /// converts float number to half precision in round-to-nearest-even mode and returns half with converted value.
        h.f = arg[i];
        /// 0x7c00 = 0111110000000000 - exponent mask
        /// s 11111 xxx xxxx xxxx - is nan (if some x is 1) or inf (if all x is 0)
        /// 0x7800 is 0111100000000000 and 0x400 is 0000010000000000
        /// number is not normal if all exponent is 1 or 0
        /// 0x7f00 is 0 11111 1100000000
        /// 0x7b00 is 0 11110 1100000000
        const bool can_round = ((h.u & 0x7F00) < 0x7B00) ? true : false;
        /// s 11111 xxx xxxx xxxx - is nan (if some x is 1) or inf (if all x is 0)
        const bool is_naninf = ((h.u & fp16_inf) == fp16_inf) ? true : false;
        /* nearest rounding masks */
        /// grs_bitmask - grs_bitmask is 0 00000 0011111111 or 0 00000 00grs11111
        unsigned short rnmask = (h.u & grs_bitmask);
        /// rne_tie - 0x180 is      0 00000 0110000000 or 384.0
        unsigned short rnmask_tie = (h.u & rne_tie);

        if (!is_naninf && can_round) {
            /* round to nearest even, if rne_mask is enabled */
            /* 0 00000 0010000000, find grs patterns */
            // 0xx - do nothing
            // 100 - this is a tie : round up if the mantissa's bit just before G is 1, else do nothing
            // 101, 110, 111 - round up > 0x0080
            h.u += (((rnmask > 0x0080) || (rnmask_tie == rne_tie)) << lshift);
        }
        h.u = (h.u & mask_mant); /* truncation */
        out[i] = h.f;
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
template <typename T>
void emulate_f8e4m3_on_fp16(const T* arg, T* out, size_t count) {
    typedef union half_t {
        unsigned short u;
        T f;
    } __half_t;

    constexpr auto use_clamp = true;
    constexpr auto exp_bits = 5;
    constexpr auto mbits = 9;
    constexpr auto non_mant_bits = exp_bits + 1;  // exponent + sign
    constexpr auto lshift = 10 - (mbits - non_mant_bits);
    constexpr auto fp16_exp_bias = 15;
    constexpr unsigned short rne_mask = 1;  // round to nearest even mask
    constexpr unsigned short mask_mant =
        static_cast<unsigned short>(0xFFFF << lshift);  /// 1111111111111111 -> 1 11111 1111000000
    constexpr unsigned short grs_bitmask = 0x007F;      /// 0 00000 0001111111
    constexpr unsigned short rne_tie = 0x00C0;          /// 0 00000 0011000000
    constexpr unsigned short fp16_inf = 0x7C00;

    __half_t h;
    for (size_t i = 0; i < count; ++i) {
        h.f = arg[i];
        float inval = ngraph::float16(arg[i]);
        /* flush values below 1-4-3 (offset=4) subnormal range to zero */
        if (fabs(inval) < 0.001953125)  // 2**-9
            h.f = 0;

        short exp_h = static_cast<short>((h.u & fp16_inf) >> 10) -
                      fp16_exp_bias;          /// 0111110000000000 -> 0000000000011111 - 15 - biased exponent for fp16
        const short sign_h = (h.u & 0x8000);  /// & 1 00000 0000000000
        short mantissa_h = (h.u & 0x03FF);    /// & 0 00000 1111111111
        ///(h.u && 0111111111111111) < 0 10010 1110000000 (19326)
        const bool can_round = ((h.u & 0x7FFF) < 0b101111110000000) ? true : false;
        bool is_naninf = ((h.u & fp16_inf) == fp16_inf) ? true : false;

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
        const unsigned short rnmask = (mantissa_h & grs_bitmask);
        /* & 0 00000 0011000000 - edge between f8e4m3 and fp16 mantissa */
        const unsigned short rnmask_tie = (mantissa_h & rne_tie);
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
        h.u = mantissa_h | sign_h;
        out[i] = h.f;
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
                       bool invert = false) {
    const auto scale_size = shape_size(scale_shape);
    const auto data_size = shape_size(data_shape);

    if (scale_size == 1) {  // per tensor scale, probably for activation
        T s = scale[0];
        T o = shift[0];
        if (invert) {
            for (size_t j = 0; j < data_size; j++) {
                out[j] = (data[j] + o) / s;
            }
        } else {
            for (size_t j = 0; j < data_size; j++) {
                out[j] = data[j] * s - o;  // o = quntized(o * s)
            }
        }
        return;
    }

    if (scale_shape[0] == 1 && scale_shape.size() > 1 && data_shape[1] == scale_shape[1] &&
        scale_size == data_shape[1]) {  // per channel scale for DW activations
        size_t step = 1;
        for (size_t i = 2; i < data_shape.size(); i++) {
            step *= data_shape[i];
        }

        for (size_t bs = 0; bs < data_shape[0]; bs++) {
            for (size_t i = 0; i < scale_size; i++) {
                T s = scale[i];
                T o = shift[i];
                if (invert) {
                    for (size_t j = 0; j < step; j++) {
                        out[j] = (data[j] + o) / s;
                    }
                } else {
                    for (size_t j = 0; j < step; j++) {
                        out[j] = data[j] * s - o;  // o = quntized(o * s)
                    }
                }
                data += step;
                out += step;
            }
        }
        return;
    }

    autobroadcast_select(data,
                         scale,
                         shift,
                         out,
                         data_shape,
                         scale_shape,
                         shift_shape,
                         op::AutoBroadcastType::NUMPY,
                         [invert](T elem, T s, T o) -> T {
                             if (invert) {
                                 return static_cast<T>((elem + o) / s);
                             }
                             return static_cast<T>(elem * s - o);
                         });
}
}  // namespace fake_convert_details

template <typename T>
bool apply_conversion(const T* data, T* out, size_t element_count, const std::string& destination_type) {
    auto inPtr = reinterpret_cast<const unsigned short*>(data);
    auto outPtr = reinterpret_cast<unsigned short*>(out);
    if (destination_type == "f8e5m2") {
        reference::fake_convert_details::emulate_f8e5m2_on_fp16(inPtr, outPtr, element_count);
    } else if (destination_type == "f8e4m3") {
        reference::fake_convert_details::emulate_f8e4m3_on_fp16(inPtr, outPtr, element_count);
    } else {
        OPENVINO_THROW("Unsupported destination type.");
    }
    return true;
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
    reference::fake_convert_details::apply_scale_shift<float16>(out,
                                                                data,
                                                                scale,
                                                                shift,
                                                                data_shape,
                                                                scale_shape,
                                                                shift_shape,
                                                                false);
    apply_conversion(out, out, element_count, destination_type);
    reference::fake_convert_details::apply_scale_shift<float16>(out,
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
    reference::fake_convert_details::apply_scale_shift<T>(out,
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

    reference::fake_convert_details::apply_scale_shift<T>(out,
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
