// Copyright (C) 2024 FUJITSU LIMITED
// SPDX-License-Identifier: Apache-2.0
//
#include <arm_sve.h>

#include "openvino/core/type/float16.hpp"

namespace ov::intel_cpu::sve_utils {

template <typename T, typename... Args>
constexpr bool one_of(T val, Args... args) {
    return ((val == args) || ...);
}

template <size_t T_SIZE>
svbool_t sve_predicate() {
    static_assert(one_of(T_SIZE, 8, 16, 32, 64), "Unexpected parameter size");
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
    static_assert(one_of(T_SIZE, 8, 16, 32, 64), "Unexpected parameter size");
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
    static_assert(one_of(T_SIZE, 8, 16, 32, 64), "Unexpected parameter size");
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
}  // namespace ov::intel_cpu::sve_utils