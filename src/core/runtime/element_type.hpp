// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Core-local element type layer. Minimal replacement for
// openvino/core/type/element_type.hpp used by the standalone Vulkan core.

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <string>

namespace ov::element {

enum class Type_t {
    undefined = 0,
    dynamic,
    boolean,
    bf16,
    f16,
    f32,
    f64,
    i4,
    i8,
    i16,
    i32,
    i64,
    u1,
    u2,
    u4,
    u8,
    u16,
    u32,
    u64,
    f4e2m1,
    f8e4m3,
    f8e5m2,
    string,
    f8e8m0,
};

class Type {
public:
    Type() = default;
    constexpr Type(const Type_t t) noexcept : m_type(t) {}
    constexpr Type_t get_type() const noexcept { return m_type; }

    constexpr operator Type_t() const noexcept { return m_type; }
    constexpr operator bool() const noexcept { return m_type != Type_t::undefined; }

    size_t size() const { return (bitwidth() + 7) >> 3; }
    size_t bitwidth() const;

    bool is_real() const {
        return m_type == Type_t::f16 || m_type == Type_t::f32 || m_type == Type_t::f64 || m_type == Type_t::bf16;
    }
    bool is_signed() const;
    bool is_quantized() const {
        return m_type == Type_t::u4 || m_type == Type_t::i4 || m_type == Type_t::u8 || m_type == Type_t::i8 ||
               m_type == Type_t::u1 || m_type == Type_t::u2;
    }
    bool is_dynamic() const { return m_type == Type_t::dynamic; }

    const char* get_type_name() const;

    friend bool operator==(const Type& a, const Type& b) { return a.m_type == b.m_type; }
    friend bool operator!=(const Type& a, const Type& b) { return a.m_type != b.m_type; }
    friend bool operator<(const Type& a, const Type& b) { return a.m_type < b.m_type; }
    friend std::ostream& operator<<(std::ostream& os, const Type& t) { return os << t.get_type_name(); }

private:
    Type_t m_type = Type_t::undefined;
};

inline constexpr Type f16{Type_t::f16};
inline constexpr Type f32{Type_t::f32};
inline constexpr Type f64{Type_t::f64};
inline constexpr Type bf16{Type_t::bf16};
inline constexpr Type i8{Type_t::i8};
inline constexpr Type i16{Type_t::i16};
inline constexpr Type i32{Type_t::i32};
inline constexpr Type i64{Type_t::i64};
inline constexpr Type u8{Type_t::u8};
inline constexpr Type u16{Type_t::u16};
inline constexpr Type u32{Type_t::u32};
inline constexpr Type u64{Type_t::u64};
inline constexpr Type i4{Type_t::i4};
inline constexpr Type u4{Type_t::u4};
inline constexpr Type boolean{Type_t::boolean};
inline constexpr Type dynamic{Type_t::dynamic};

}  // namespace ov::element

namespace ov {

// 16-bit half precision float. Enough surface for the core: bit
// reconstruction and numeric_limits (max/min) for data_type_traits.
class float16 {
public:
    float16() = default;
    constexpr explicit float16(uint16_t bits) : m_bits(bits) {}
    static float16 from_bits(uint16_t bits) { return float16(bits); }
    uint16_t to_bits() const { return m_bits; }

    friend bool operator==(const float16& a, const float16& b) { return a.m_bits == b.m_bits; }
    friend bool operator!=(const float16& a, const float16& b) { return a.m_bits != b.m_bits; }

private:
    uint16_t m_bits = 0;
};

}  // namespace ov

namespace std {

template <>
struct numeric_limits<ov::float16> {
    static constexpr bool is_specialized = true;
    static ov::float16 max() noexcept { return ov::float16::from_bits(0x7BFF); }     // 65504
    static ov::float16 lowest() noexcept { return ov::float16::from_bits(0xFBFF); }  // -65504
    static ov::float16 min() noexcept { return ov::float16::from_bits(0x0400); }      // 6.1035e-5
    static ov::float16 epsilon() noexcept { return ov::float16::from_bits(0x1400); }  // 9.7656e-4
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr int digits = 11;
};

}  // namespace std
