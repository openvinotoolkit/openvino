// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/fake_convert.hpp"

#include <cmath>

#include "openvino/core/type/float8_e4m3.hpp"

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
void emulate_f8e5m2_on_fp16(const float16* const arg_f, float16* out_f, size_t count, bool use_clamp) {
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
    constexpr uint16_t fp16_mant = 0x03FF;
    constexpr uint16_t fp16_quiet_nan_bit = 0x0200;

    for (size_t i = 0; i < count; ++i) {
        /// converts float number to half precision in round-to-nearest-even mode and returns half with converted value.
        val_bit_repr = arg_u[i];
        /// s 11111 xxx xxxx xxxx - is nan (if some x is 1) or inf (if all x is 0)
        const bool is_naninf = ((val_bit_repr & fp16_inf) == fp16_inf) ? true : false;
        /* nearest rounding masks */
        /// grs_bitmask - grs_bitmask is 0 00000 0011111111 or 0 00000 00grs11111
        uint16_t rnmask = (val_bit_repr & grs_bitmask);
        /// rne_tie - 0x180 is      0 00000 0110000000 or 384.0
        uint16_t rnmask_tie = (val_bit_repr & rne_tie);

        if (!is_naninf) {
            /* round to nearest even, if rne_mask is enabled */
            /* 0 00000 0010000000, find grs patterns */
            // 0xx - do nothing
            // 100 - this is a tie : round up if the mantissa's bit just before G is 1, else do nothing
            // 101, 110, 111 - round up > 0x0080
            val_bit_repr += (((rnmask > 0x0080) || (rnmask_tie == rne_tie)) << lshift);
        } else if ((val_bit_repr & fp16_mant) != 0) {
            /* set quiet bit of NaN, so it stays NaN (not inf) after mantissa truncation */
            val_bit_repr |= fp16_quiet_nan_bit;
        }
        val_bit_repr &= mask_mant; /* truncation */
        if (use_clamp) {
            // clamp inf to max and -inf to lowest, S.11111.00 -> S.11110.11
            val_bit_repr -= (((val_bit_repr & 0x7F00) == fp16_inf) << lshift);
        }
        out_u[i] = val_bit_repr;
    }
}

/**
 * @brief Emulation of conversion fp16 value to f8e4m3 format
 *
 * @param arg_f       Pointer to the input data.
 * @param out_f       Pointer to the otuput data.
 * @param count       Number of elements in the data input.
 * @param use_clamp   If use clamp. f8e4m3 has no infinity, so without clamp values out of range (and inf) become NaN,
 *                    with clamp finite and infinite values saturate to +-448. NaN always stays NaN.
 *
 * The conversion is done by float8_e4m3 which rounds to nearest even, including subnormal range and keeps the sign
 * of zero. The f16 -> f32 -> f16 steps used on the way are exact, so no double rounding happens.
 */
void emulate_f8e4m3_on_fp16(const float16* arg_f, float16* out_f, size_t count, bool use_clamp) {
    constexpr float f8e4m3_max = 448.0f;

    for (size_t i = 0; i < count; ++i) {
        const float val = static_cast<float>(arg_f[i]);
        if (std::isnan(val)) {
            out_f[i] = arg_f[i];
            continue;
        }
        const float8_e4m3 f8(val);
        float res = static_cast<float>(f8);
        if (std::isnan(res) && use_clamp) {
            // overflow (or inf) which is not NaN on input: saturate
            res = std::signbit(val) ? -f8e4m3_max : f8e4m3_max;
        }
        out_f[i] = float16(res);
    }
}
}  // namespace func

namespace fake_convert_details {
/**
 * @brief Call conversion of fp16 value to the desired destination type
 *
 * @param arg                  Pointer to the input data.
 * @param out                  Pointer to the otuput data.
 * @param count                Number of elements in the data input.
 * @param destination_type     Name of the destination type.
 */
void apply_conversion(const float16* data, float16* out, size_t element_count, const element::Type& destination_type) {
    if (destination_type == element::f8e5m2) {
        reference::func::emulate_f8e5m2_on_fp16(data, out, element_count);
    } else if (destination_type == element::f8e4m3) {
        reference::func::emulate_f8e4m3_on_fp16(data, out, element_count);
    } else {
        OPENVINO_THROW("Unsupported destination type.");
    }
}
}  // namespace fake_convert_details
}  // namespace reference
}  // namespace ov
