// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov {
namespace reference {
namespace detail {
inline void set_u1(uint8_t* buf, size_t idx, uint8_t val) {
    const size_t byte_idx = idx / 8;
    const uint8_t bit_idx = 7 - (idx % 8);
    if (val) {
        buf[byte_idx] |= (1 << bit_idx);
    } else {
        buf[byte_idx] &= ~(1 << bit_idx);
    }
}

inline uint8_t get_u1(const uint8_t* buf, size_t idx) {
    const size_t byte_idx = idx / 8;
    const uint8_t bit_idx = 7 - (idx % 8);
    return (buf[byte_idx] & (1 << bit_idx)) ? 1 : 0;
}

inline void set_u4(uint8_t* buf, size_t idx, uint8_t val) {
    const size_t byte_idx = idx / 2;
    const uint8_t bit_shift = 4 * (++idx % 2);
    buf[byte_idx] &= ~(0xF << bit_shift);         // half byte zeroed
    buf[byte_idx] |= ((val & 0xF) << bit_shift);  // set 1's
}

inline uint8_t get_u4(const uint8_t* buf, size_t idx) {
    const size_t byte_idx = idx / 2;
    const uint8_t bit_shift = 4 * (++idx % 2);
    return (buf[byte_idx] >> bit_shift) & 0xF;
}

inline void set_i4(uint8_t* buf, size_t idx, int8_t val) {
    const size_t byte_idx = idx / 2;
    const uint8_t bit_shift = 4 * (++idx % 2);
    buf[byte_idx] &= ~(0xF << bit_shift);         // half byte zeroed
    buf[byte_idx] |= ((val & 0xF) << bit_shift);  // set 1's
}

inline int8_t get_i4(const uint8_t* buf, size_t idx) {
    const size_t byte_idx = idx / 2;
    const uint8_t bit_shift = 4 * (++idx % 2);
    uint8_t val = (buf[byte_idx] >> bit_shift) & 0xF;
    if (val & 0x08) {  // negative number
        val |= 0xF0;
    }
    return val;
}
template <typename TO, typename TI>
TO get_value(const uint8_t* buf, size_t idx, element::Type from_type) {
    if (from_type == element::u1) {
        return detail::get_u1(buf, idx);
    }

    if (from_type == element::u4) {
        return detail::get_u4(buf, idx);
    }

    if (from_type == element::i4) {
        return detail::get_i4(buf, idx);
    }

    auto v = reinterpret_cast<const TI*>(buf);
    return static_cast<TO>(v[idx]);
}

template <typename TI, typename TO>
void lp_convert(const TI* arg, TO* out, size_t count, element::Type_t src_type, element::Type_t dst_type) {
    const uint8_t* input = reinterpret_cast<const uint8_t*>(arg);
    uint8_t* output = reinterpret_cast<uint8_t*>(out);
    for (size_t i = 0; i < count; ++i) {
        if (dst_type == element::u1) {
            detail::set_u1(output, i, detail::get_value<uint8_t, TI>(input, i, src_type));
        } else if (dst_type == element::u4) {
            detail::set_u4(output, i, detail::get_value<uint8_t, TI>(input, i, src_type));
        } else if (dst_type == element::i4) {
            detail::set_i4(output, i, detail::get_value<int8_t, TI>(input, i, src_type));
        } else if (src_type == element::nf4) {
            ConvertNF4::unpack(out, input, i);
        } else {
            out[i] = detail::get_value<TO, TI>(input, i, src_type);
        }
    }
}
}  // namespace detail

template <typename TI, typename TO>
typename std::enable_if<!std::is_same<TO, char>::value>::type convert(const TI* arg, TO* out, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<TO>(arg[i]);
    }
}

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)

template <>
void convert<uint8_t, float16>(const uint8_t* arg, float16* out, size_t count);
template <>
void convert<float16, float>(const float16* arg, float* out, size_t count);
template <>
void convert<float, float16>(const float* arg, float16* out, size_t count);
template <>
void convert<float, int8_t>(const float* arg, int8_t* out, size_t count);
template <>
void convert<float16, int8_t>(const float16* arg, int8_t* out, size_t count);

#endif  // OPENVINO_ARCH_X86 || OPENVINO_ARCH_X86_64

// Count how many f32 values is out of normal finite numbers range when converted to f16
size_t count_out_of_f16_range(const float* arg, size_t count);

// Convert values from f32 to f16 with claming to f16 min/max when value is out of normal finite numbers range
void convert_from_f32_to_f16_with_clamp(const float* arg, float16* out, size_t count);

// overload to handle ov::boolean (it is stored as char)
template <typename TI, typename TO>
typename std::enable_if<std::is_same<TO, char>::value>::type convert(const TI* arg, TO* out, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<char>(static_cast<bool>(arg[i]));
    }
}

}  // namespace reference
}  // namespace ov
