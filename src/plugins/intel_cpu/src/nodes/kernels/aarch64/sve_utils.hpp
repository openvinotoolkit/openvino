// Copyright (C) 2024 FUJITSU LIMITED
// SPDX-License-Identifier: Apache-2.0
//
#include <arm_sve.h>

#include "openvino/core/type/float16.hpp"

namespace ov::intel_cpu::sve_utils {

template <typename T, typename... Args>
constexpr bool any_of(T val, Args... args) {
    return ((val == args) || ...);
}

template <size_t T_SIZE>
svbool_t sve_predicate() {
    static_assert(any_of(T_SIZE, 8, 16, 32, 64), "Unexpected parameter size");
    if constexpr (8 == T_SIZE) {
        return svptrue_b8();
    } else if (16 == T_SIZE) {
        return svptrue_b16();
    } else if (32 == T_SIZE) {
        return svptrue_b32();
    } else if (64 == T_SIZE) {
        return svptrue_b64();
    }
}

template <typename T_TYPE, size_t T_SIZE>
svbool_t sve_predicate(T_TYPE lower, T_TYPE higher) {
    static_assert(any_of(T_SIZE, 8, 16, 32, 64), "Unexpected parameter size");
    if constexpr (8 == T_SIZE) {
        return svwhilelt_b8(lower, higher);
    } else if (16 == T_SIZE) {
        return svwhilelt_b16(lower, higher);
    } else if (32 == T_SIZE) {
        return svwhilelt_b32(lower, higher);
    } else if (64 == T_SIZE) {
        return svwhilelt_b64(lower, higher);
    }
}

template <size_t T_SIZE>
size_t sve_vlen() {
    static_assert(any_of(T_SIZE, 8, 16, 32, 64), "Unexpected parameter size");
    if constexpr (8 == T_SIZE) {
        return svcntb();
    } else if (16 == T_SIZE) {
        return svcnth();
    } else if (32 == T_SIZE) {
        return svcntw();
    } else if (64 == T_SIZE) {
        return svcntd();
    }
}

template <typename TA, typename TB>
static void cvt_copy(TA* dst, TB* src, size_t n) {
    size_t i = 0;
    if constexpr (std::is_same<TA, TB>::value) {
        auto pg_dst = sve_predicate<sizeof(TA)>();
        auto vlen = sve_vlen<sizeof(TA)>();
        for (; i + vlen <= n; i += vlen) {
            auto vb = svld1(pg_dst, src + i);
            svst1(pg_dst, dst + i, vb);
        }
        auto pgt = sve_predicate<TA, sizeof(TA)>(i, n);
        auto vb = svld1(pg_dst, src + i);
        svst1(pg_dst, dst + i, vb);
        return;
    } else if constexpr (std::is_same<TA, float>::value && std::is_same<TB, ov::float16>::value) {
        auto src_ptr = reinterpret_cast<float16_t*>(src);
        auto pg_vl2 = svwhilelt_b16(svcnth() / 2, svcnth());
        auto vlen = svcnth() / 2;
        auto pg_dst = svptrue_b32();
        for (; i + vlen <= n; i += vlen) {
            auto load_src = svld1_f16(pg_vl2, src_ptr + i);
            auto src_interleave = svzip1_f16(load_src, load_src);
            auto cvt_dst = svcvt_f32_f16_z(pg_dst, src_interleave);
            svst1(pg_dst, dst + i, cvt_dst);
        }
    }
}

}  // namespace ov::intel_cpu::sve_utils