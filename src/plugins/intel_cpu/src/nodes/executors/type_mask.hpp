// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <limits>

#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace intel_cpu {
struct TypeMask {
    enum Value : uint64_t {
        _undefined = 1 << 0,
        _dynamic   = 1 << 1,
        _boolean   = 1 << 2,
        _bf16      = 1 << 3,
        _f16       = 1 << 4,
        _f32       = 1 << 5,
        _f64       = 1 << 6,
        _i4        = 1 << 7,
        _i8        = 1 << 8,
        _i16       = 1 << 9,
        _i32       = 1 << 10,
        _i64       = 1 << 11,
        _u1        = 1 << 12,
        _u4        = 1 << 13,
        _u8        = 1 << 14,
        _u16       = 1 << 15,
        _u32       = 1 << 16,
        _u64       = 1 << 17,
        _nf4       = 1 << 18,
        _f8e4m3    = 1 << 19,
        _f8e5m2    = 1 << 20,
        _string    = 1 << 21,
        _f4e2m1    = 1 << 22,
        _f8e8m0    = 1 << 23,
    };

    TypeMask(const ov::element::Type precision) : value(generateMask(precision)), precision(precision) {}

    TypeMask(const uint64_t value) : value(value) {}

    // can be treated as uint64_t mask
    operator uint64_t() const {
        return value;
    }
    // match
    bool operator&(const ov::element::Type precision) const {
        return value & TypeMask(precision);
    }

    const uint64_t value;
    const ov::element::Type precision;

private:
    static Value generateMask(const ov::element::Type type) {
#define CASE(typeM)          \
    case ov::element::typeM: \
        return _##typeM;
        switch (type) {
            CASE(undefined)
            CASE(dynamic)
            CASE(boolean)
            CASE(bf16)
            CASE(f16)
            CASE(f32)
            CASE(f64)
            CASE(i4)
            CASE(i8)
            CASE(i16)
            CASE(i32)
            CASE(i64)
            CASE(u1)
            CASE(u4)
            CASE(u8)
            CASE(u16)
            CASE(u32)
            CASE(u64)
            CASE(nf4)
            CASE(f8e4m3)
            CASE(f8e5m2)
            CASE(string)
            CASE(f4e2m1)
            CASE(f8e8m0)
        default:
            return _undefined;
        }
#undef CASE
    }
};

namespace TypeMaskAlias {
constexpr ov::element::Type fxx(ov::element::Type_t::undefined);
#define DEFINE_TYPE_ALIAS(x) constexpr auto x = TypeMask::Value::x
// use underscore for naming to avoid conflicts with Precision aliases
DEFINE_TYPE_ALIAS(_undefined);
DEFINE_TYPE_ALIAS(_dynamic);
DEFINE_TYPE_ALIAS(_boolean);
DEFINE_TYPE_ALIAS(_bf16);
DEFINE_TYPE_ALIAS(_f16);
DEFINE_TYPE_ALIAS(_f32);
DEFINE_TYPE_ALIAS(_f64);
DEFINE_TYPE_ALIAS(_i4);
DEFINE_TYPE_ALIAS(_i8);
DEFINE_TYPE_ALIAS(_i16);
DEFINE_TYPE_ALIAS(_i32);
DEFINE_TYPE_ALIAS(_i64);
DEFINE_TYPE_ALIAS(_u1);
DEFINE_TYPE_ALIAS(_u4);
DEFINE_TYPE_ALIAS(_u8);
DEFINE_TYPE_ALIAS(_u16);
DEFINE_TYPE_ALIAS(_u32);
DEFINE_TYPE_ALIAS(_u64);
DEFINE_TYPE_ALIAS(_nf4);
DEFINE_TYPE_ALIAS(_f8e4m3);
DEFINE_TYPE_ALIAS(_f8e5m2);
DEFINE_TYPE_ALIAS(_string);
DEFINE_TYPE_ALIAS(_f4e2m1);
DEFINE_TYPE_ALIAS(_f8e8m0);
constexpr auto _any_float = _f64 | _f32 | _f16 | _bf16;
constexpr auto _half_float = _f16 | _bf16;
constexpr auto _quant = _u8 | _i8;
constexpr auto _any = std::numeric_limits<uint64_t>::max();
}  // namespace TypeMaskAlias

}  // namespace intel_cpu
}  // namespace ov
