// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Scalar ISA specializations for simd::vec, simd::mask, and free functions.
// No intrinsics — pure C++. Always available as fallback.

#pragma once

#include <cassert>
#include <type_traits>

#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "simd_common.hpp"

namespace ov::Extensions::Cpu::XARCH::simd {

// --- vec<float> ------------------------------------------------------------

template <>
struct vec<float, isa::scalar> {
    using element_type = float;
    static constexpr int width = 1;
    static constexpr isa isa_value = isa::scalar;
    float v;

    vec() : v(0.0F) {}
    vec(float val) : v(val) {}  // NOLINT(google-explicit-constructor)

    vec operator+(vec b) const {
        return {v + b.v};
    }
    vec operator-(vec b) const {
        return {v - b.v};
    }
    vec operator*(vec b) const {
        return {v * b.v};
    }
};

inline void store(vec<float, isa::scalar> v, float* p) {
    *p = v.v;
}
inline void store(vec<float, isa::scalar> v, ov::bfloat16* p) {
    *p = ov::bfloat16(v.v);
}
inline void store(vec<float, isa::scalar> v, ov::float16* p) {
    *p = ov::float16(v.v);
}
inline float reduce(vec<float, isa::scalar> v) {
    return v.v;
}
inline vec<float, isa::scalar> fmadd(vec<float, isa::scalar> a, vec<float, isa::scalar> b, vec<float, isa::scalar> c) {
    return {a.v * b.v + c.v};
}

// --- vec<int32_t> ----------------------------------------------------------

template <>
struct vec<int32_t, isa::scalar> {
    using element_type = int32_t;
    static constexpr int width = 1;
    static constexpr isa isa_value = isa::scalar;
    int32_t v;

    vec() : v(0) {}
    vec(int32_t val) : v(val) {}  // NOLINT(google-explicit-constructor)

    vec operator&(vec b) const {
        return {v & b.v};
    }
};

inline void store(vec<int32_t, isa::scalar> v, int32_t* p) {
    *p = v.v;
}

// --- loads -----------------------------------------------------------------

inline vec<float, isa::scalar> load(const float* p, vec<float, isa::scalar>* /*tag*/) {
    return {*p};
}
inline vec<float, isa::scalar> load(const ov::float16* p, vec<float, isa::scalar>* /*tag*/) {
    return {static_cast<float>(*p)};
}
inline vec<float, isa::scalar> load(const ov::bfloat16* p, vec<float, isa::scalar>* /*tag*/) {
    return {static_cast<float>(*p)};
}
inline vec<float, isa::scalar> load(const uint8_t* p, vec<float, isa::scalar>* /*tag*/) {
    return {static_cast<float>(*p)};
}
inline vec<float, isa::scalar> partial_load(uint32_t k, const float* p, vec<float, isa::scalar>* /*tag*/) {
    return {(k & 1) ? *p : 0.0F};
}
inline vec<float, isa::scalar> load_u4(const uint8_t* p, int bit_offset, vec<float, isa::scalar>* /*tag*/) {
    return {static_cast<float>((*p >> (4 - bit_offset)) & 0x0F)};
}
inline void load_u4_pair(const uint8_t* p, vec<float, isa::scalar>& lo, vec<float, isa::scalar>& hi) {
    lo.v = static_cast<float>(*p & 0x0F);
    hi.v = static_cast<float>((*p >> 4) & 0x0F);
}
inline void load_u8_pair(const uint8_t* p, vec<float, isa::scalar>& lo, vec<float, isa::scalar>& hi) {
    lo = load(p, static_cast<vec<float, isa::scalar>*>(nullptr));
    hi = load(p + 1, static_cast<vec<float, isa::scalar>*>(nullptr));
}

inline vec<int32_t, isa::scalar> load(const int32_t* p, vec<int32_t, isa::scalar>* /*tag*/) {
    return {*p};
}
inline vec<int32_t, isa::scalar> load(const uint8_t* p, vec<int32_t, isa::scalar>* /*tag*/) {
    return {static_cast<int32_t>(*p)};
}

// --- arithmetic / permute --------------------------------------------------

inline vec<int32_t, isa::scalar> srlv(vec<int32_t, isa::scalar> val, vec<int32_t, isa::scalar> shift) {
    return {static_cast<int32_t>(static_cast<uint32_t>(val.v) >> shift.v)};
}

inline vec<float, isa::scalar> permute(vec<float, isa::scalar> table, vec<int32_t, isa::scalar> idx) {
    (void)idx;
    return table;
}

inline vec<float, isa::scalar> permute2(vec<float, isa::scalar> table_lo,
                                        vec<int32_t, isa::scalar> idx,
                                        vec<float, isa::scalar> table_hi) {
    return (idx.v & 1) == 0 ? table_lo : table_hi;
}

// Scalar stubs for shuffle operations — satisfy template name lookup only.
inline vec<float, isa::scalar> unpack_lo(vec<float, isa::scalar> /*a*/, vec<float, isa::scalar> /*b*/) {
    assert(false && "unpack_lo: scalar stub");  // NOLINT
    return {};
}
inline vec<float, isa::scalar> unpack_hi(vec<float, isa::scalar> /*a*/, vec<float, isa::scalar> /*b*/) {
    assert(false && "unpack_hi: scalar stub");  // NOLINT
    return {};
}
inline vec<float, isa::scalar> unpack_lo_64(vec<float, isa::scalar> /*a*/, vec<float, isa::scalar> /*b*/) {
    assert(false && "unpack_lo_64: scalar stub");  // NOLINT
    return {};
}
inline vec<float, isa::scalar> unpack_hi_64(vec<float, isa::scalar> /*a*/, vec<float, isa::scalar> /*b*/) {
    assert(false && "unpack_hi_64: scalar stub");  // NOLINT
    return {};
}
// Always-false-when-instantiated helper (depends on template param so it's only
// evaluated at instantiation, not at template definition).
template <int>
struct scalar_always_false : std::false_type {};

template <int ctrl>
inline vec<float, isa::scalar> shuffle(vec<float, isa::scalar> /*a*/, vec<float, isa::scalar> /*b*/) {
    static_assert(scalar_always_false<ctrl>::value, "shuffle: not available for scalar ISA");
    return {};
}
template <int ctrl>
inline vec<float, isa::scalar> permute_lanes(vec<float, isa::scalar> /*a*/, vec<float, isa::scalar> /*b*/) {
    static_assert(scalar_always_false<ctrl>::value, "permute_lanes: not available for scalar ISA");
    return {};
}
template <int ctrl>
inline vec<float, isa::scalar> permute_64(vec<float, isa::scalar> /*a*/) {
    static_assert(scalar_always_false<ctrl>::value, "permute_64: not available for scalar ISA");
    return {};
}

// --- mask ------------------------------------------------------------------

template <>
struct mask<isa::scalar> {
    bool v;
    mask() : v(false) {}
    mask(bool val) : v(val) {}  // NOLINT(google-explicit-constructor)
};

inline mask<isa::scalar> operator>(vec<int32_t, isa::scalar> a, vec<int32_t, isa::scalar> b) {
    return {a.v > b.v};
}
inline mask<isa::scalar> operator<(vec<int32_t, isa::scalar> a, vec<int32_t, isa::scalar> b) {
    return {a.v < b.v};
}
inline mask<isa::scalar> operator>=(vec<int32_t, isa::scalar> a, vec<int32_t, isa::scalar> b) {
    return {a.v >= b.v};
}
inline mask<isa::scalar> operator<=(vec<int32_t, isa::scalar> a, vec<int32_t, isa::scalar> b) {
    return {a.v <= b.v};
}
inline mask<isa::scalar> operator==(vec<int32_t, isa::scalar> a, vec<int32_t, isa::scalar> b) {
    return {a.v == b.v};
}
inline mask<isa::scalar> operator!=(vec<int32_t, isa::scalar> a, vec<int32_t, isa::scalar> b) {
    return {a.v != b.v};
}

inline vec<float, isa::scalar> select(mask<isa::scalar> m,
                                      vec<float, isa::scalar> if_false,
                                      vec<float, isa::scalar> if_true) {
    return m.v ? if_true : if_false;
}

}  // namespace ov::Extensions::Cpu::XARCH::simd
