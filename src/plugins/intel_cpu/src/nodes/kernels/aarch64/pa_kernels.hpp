// Copyright (C) 2024 FUJITSU LIMITED
// SPDX-License-Identifier: Apache-2.0
//
#include <arm_sve.h>

#include "openvino/core/type/float16.hpp"

#define SIZE_IN_BITS(t_var) sizeof(t_var) * 8
#define __ce(expr, bits, ...)     \
    if constexpr (expr == bits) { \
        __VA_ARGS__               \
    }

#define SVE_PREDICATE(var, t_var)                                                                         \
    svbool_t var;                                                                                         \
                                                                                                          \
    __ce(SIZE_IN_BITS(t_var), 8, var = svptrue_b8();) __ce(SIZE_IN_BITS(t_var), 16, var = svptrue_b16();) \
        __ce(SIZE_IN_BITS(t_var), 32, var = svptrue_b32();) __ce(SIZE_IN_BITS(t_var), 64, var = svptrue_b64();)

#define SVE_VLEN(var, t_var)                                                                     \
    size_t var;                                                                                  \
                                                                                                 \
    __ce(SIZE_IN_BITS(t_var), 8, var = svcntb();) __ce(SIZE_IN_BITS(t_var), 16, var = svcnth();) \
        __ce(SIZE_IN_BITS(t_var), 32, var = svcntw();) __ce(SIZE_IN_BITS(t_var), 64, var = svcntd();)

#define SVE_PREDICATE_WHILELT(var, t_var, arg1, arg2)                       \
    svbool_t var;                                                           \
                                                                            \
    __ce(SIZE_IN_BITS(t_var), 8, var = svwhilelt_b8(arg1, arg2);)           \
        __ce(SIZE_IN_BITS(t_var), 16, var = svwhilelt_b16(arg1, arg2);)     \
            __ce(SIZE_IN_BITS(t_var), 32, var = svwhilelt_b32(arg1, arg2);) \
                __ce(SIZE_IN_BITS(t_var), 64, var = svwhilelt_b64(arg1, arg2);)

namespace ov::Extensions::Cpu::XARCH {
static void cvt_copy(float* dst, ov::float16* src, size_t n) {
    auto src_ptr = reinterpret_cast<float16_t*>(src);
    auto pg_vl2 = svwhilelt_b16(svcnth() / 2, svcnth());
    auto vlen = svcnth() / 2;
    auto pg_dst = svptrue_b32();
    size_t i = 0;
    for (; i + vlen <= n; i += vlen) {
        auto load_src = svld1_f16(pg_vl2, src_ptr + i);
        auto src_interleave = svzip1_f16(load_src, load_src);
        auto cvt_dst = svcvt_f32_f16_z(pg_dst, src_interleave);
        svst1(pg_dst, dst + i, cvt_dst);
    }
    for (; i < n; i++) {
        dst[i] = src[i];
    }
}

static void cvt_copy(ov::float16* dst, float* src, size_t n) {
    auto dst_ptr = reinterpret_cast<float16_t*>(dst);
    auto pg_src = svptrue_b32();
    auto pg_dst = svwhilelt_b16(svcnth() / 2, svcnth());
    auto vlen = svcntw();
    size_t i = 0;
    for (; i + vlen < n; i += vlen) {
        auto load_src = svld1_f32(pg_src, src + i);
        auto cvt_dst = svcvt_f16_f32_z(pg_src, load_src);
        auto str_dst = svuzp1(cvt_dst, cvt_dst);
        svst1_f16(pg_dst, dst_ptr + i, str_dst);
    }
    for (; i < n; i++) {
        dst[i] = src[i];
    }
}
}  // namespace ov::Extensions::Cpu::XARCH