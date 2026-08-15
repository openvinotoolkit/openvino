// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Core-local element type implementation (element_type.hpp).

#include "element_type.hpp"

namespace ov::element {

size_t Type::bitwidth() const {
    switch (m_type) {
    case Type_t::boolean:
        return 8;
    case Type_t::bf16:
    case Type_t::f16:
        return 16;
    case Type_t::f32:
    case Type_t::i32:
    case Type_t::u32:
        return 32;
    case Type_t::f64:
    case Type_t::i64:
    case Type_t::u64:
        return 64;
    case Type_t::i4:
    case Type_t::u4:
        return 4;
    case Type_t::i8:
    case Type_t::u8:
        return 8;
    case Type_t::i16:
    case Type_t::u16:
        return 16;
    case Type_t::u1:
        return 1;
    case Type_t::u2:
        return 2;
    case Type_t::f4e2m1:
        return 4;
    case Type_t::f8e4m3:
    case Type_t::f8e5m2:
    case Type_t::f8e8m0:
        return 8;
    case Type_t::undefined:
    case Type_t::dynamic:
    case Type_t::string:
    default:
        return 0;
    }
}

bool Type::is_signed() const {
    switch (m_type) {
    case Type_t::i4:
    case Type_t::i8:
    case Type_t::i16:
    case Type_t::i32:
    case Type_t::i64:
        return true;
    default:
        return false;
    }
}

const char* Type::get_type_name() const {
    switch (m_type) {
    case Type_t::undefined:
        return "undefined";
    case Type_t::dynamic:
        return "dynamic";
    case Type_t::boolean:
        return "boolean";
    case Type_t::bf16:
        return "bf16";
    case Type_t::f16:
        return "f16";
    case Type_t::f32:
        return "f32";
    case Type_t::f64:
        return "f64";
    case Type_t::i4:
        return "i4";
    case Type_t::i8:
        return "i8";
    case Type_t::i16:
        return "i16";
    case Type_t::i32:
        return "i32";
    case Type_t::i64:
        return "i64";
    case Type_t::u1:
        return "u1";
    case Type_t::u2:
        return "u2";
    case Type_t::u4:
        return "u4";
    case Type_t::u8:
        return "u8";
    case Type_t::u16:
        return "u16";
    case Type_t::u32:
        return "u32";
    case Type_t::u64:
        return "u64";
    case Type_t::f4e2m1:
        return "f4e2m1";
    case Type_t::f8e4m3:
        return "f8e4m3";
    case Type_t::f8e5m2:
        return "f8e5m2";
    case Type_t::f8e8m0:
        return "f8e8m0";
    case Type_t::string:
        return "string";
    default:
        return "undefined";
    }
}

}  // namespace ov::element
