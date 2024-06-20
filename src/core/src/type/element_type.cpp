// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/element_type.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <unordered_map>

#include "openvino/core/type/element_type_traits.hpp"

namespace {
struct TypeInfo {
    size_t m_bitwidth;
    bool m_is_real;
    bool m_is_signed;
    bool m_is_quantized;
    const char* m_cname;
    const char* m_type_name;
};

struct ElementTypes {
    struct TypeHash {
        size_t operator()(ov::element::Type_t t) const {
            return static_cast<size_t>(t);
        }
    };

    using ElementsMap = std::unordered_map<ov::element::Type_t, TypeInfo, TypeHash>;
};

inline TypeInfo get_type_info(ov::element::Type_t type) {
    switch (type) {
    case ov::element::Type_t::undefined:
        return {std::numeric_limits<size_t>::max(), false, false, false, "undefined", "undefined"};
    case ov::element::Type_t::dynamic:
        return {0, false, false, false, "dynamic", "dynamic"};
    case ov::element::Type_t::boolean:
        return {8, false, true, false, "char", "boolean"};
    case ov::element::Type_t::bf16:
        return {16, true, true, false, "bfloat16", "bf16"};
    case ov::element::Type_t::f16:
        return {16, true, true, false, "float16", "f16"};
    case ov::element::Type_t::f32:
        return {32, true, true, false, "float", "f32"};
    case ov::element::Type_t::f64:
        return {64, true, true, false, "double", "f64"};
    case ov::element::Type_t::i4:
        return {4, false, true, true, "int4_t", "i4"};
    case ov::element::Type_t::i8:
        return {8, false, true, true, "int8_t", "i8"};
    case ov::element::Type_t::i16:
        return {16, false, true, false, "int16_t", "i16"};
    case ov::element::Type_t::i32:
        return {32, false, true, true, "int32_t", "i32"};
    case ov::element::Type_t::i64:
        return {64, false, true, false, "int64_t", "i64"};
    case ov::element::Type_t::u1:
        return {1, false, false, false, "uint1_t", "u1"};
    case ov::element::Type_t::u2:
        return {2, false, false, false, "uint2_t", "u2"};
    case ov::element::Type_t::u3:
        return {3, false, false, false, "uint3_t", "u3"};
    case ov::element::Type_t::u4:
        return {4, false, false, false, "uint4_t", "u4"};
    case ov::element::Type_t::u6:
        return {6, false, false, false, "uint6_t", "u6"};
    case ov::element::Type_t::u8:
        return {8, false, false, true, "uint8_t", "u8"};
    case ov::element::Type_t::u16:
        return {16, false, false, false, "uint16_t", "u16"};
    case ov::element::Type_t::u32:
        return {32, false, false, false, "uint32_t", "u32"};
    case ov::element::Type_t::u64:
        return {64, false, false, false, "uint64_t", "u64"};
    case ov::element::Type_t::nf4:
        return {4, false, false, true, "nfloat4", "nf4"};
    case ov::element::Type_t::f8e4m3:
        return {8, true, true, true, "f8e4m3", "f8e4m3"};
    case ov::element::Type_t::f8e5m2:
        return {8, true, true, true, "f8e5m2", "f8e5m2"};
    case ov::element::Type_t::string:
        return {8 * sizeof(std::string), false, false, false, "string", "string"};
    case ov::element::Type_t::f4e2m1:
        return {4, true, true, true, "f4e2m1", "f4e2m1"};
    case ov::element::Type_t::f8e8m0:
        return {8, true, true, true, "f8e8m0", "f8e8m0"};
    default:
        OPENVINO_THROW("ov::element::Type_t not supported: ", type);
    }
};

ov::element::Type type_from_string(const std::string& type) {
    if (type == "f16" || type == "FP16") {
        return ::ov::element::Type(::ov::element::Type_t::f16);
    } else if (type == "f32" || type == "FP32") {
        return ::ov::element::Type(::ov::element::Type_t::f32);
    } else if (type == "bf16" || type == "BF16") {
        return ::ov::element::Type(::ov::element::Type_t::bf16);
    } else if (type == "f64" || type == "FP64") {
        return ::ov::element::Type(::ov::element::Type_t::f64);
    } else if (type == "i4" || type == "I4") {
        return ::ov::element::Type(::ov::element::Type_t::i4);
    } else if (type == "i8" || type == "I8") {
        return ::ov::element::Type(::ov::element::Type_t::i8);
    } else if (type == "i16" || type == "I16") {
        return ::ov::element::Type(::ov::element::Type_t::i16);
    } else if (type == "i32" || type == "I32") {
        return ::ov::element::Type(::ov::element::Type_t::i32);
    } else if (type == "i64" || type == "I64") {
        return ::ov::element::Type(::ov::element::Type_t::i64);
    } else if (type == "u1" || type == "U1" || type == "BIN" || type == "bin") {
        return ::ov::element::Type(::ov::element::Type_t::u1);
    } else if (type == "u2" || type == "U2") {
        return ::ov::element::Type(::ov::element::Type_t::u2);
    } else if (type == "u3" || type == "U3") {
        return ::ov::element::Type(::ov::element::Type_t::u3);
    } else if (type == "u4" || type == "U4") {
        return ::ov::element::Type(::ov::element::Type_t::u4);
    } else if (type == "u6" || type == "U6") {
        return ::ov::element::Type(::ov::element::Type_t::u6);
    } else if (type == "u8" || type == "U8") {
        return ::ov::element::Type(::ov::element::Type_t::u8);
    } else if (type == "u16" || type == "U16") {
        return ::ov::element::Type(::ov::element::Type_t::u16);
    } else if (type == "u32" || type == "U32") {
        return ::ov::element::Type(::ov::element::Type_t::u32);
    } else if (type == "u64" || type == "U64") {
        return ::ov::element::Type(::ov::element::Type_t::u64);
    } else if (type == "boolean" || type == "BOOL") {
        return ::ov::element::Type(::ov::element::Type_t::boolean);
    } else if (type == "string" || type == "STRING") {
        return ::ov::element::Type(::ov::element::Type_t::string);
    } else if (type == "undefined" || type == "UNSPECIFIED") {
        return ::ov::element::Type(::ov::element::Type_t::undefined);
    } else if (type == "dynamic") {
        return ::ov::element::Type(::ov::element::Type_t::dynamic);
    } else if (type == "nf4" || type == "NF4") {
        return ::ov::element::Type(::ov::element::Type_t::nf4);
    } else if (type == "f8e4m3" || type == "F8E4M3") {
        return ::ov::element::Type(::ov::element::Type_t::f8e4m3);
    } else if (type == "f8e5m2" || type == "F8E5M2") {
        return ::ov::element::Type(::ov::element::Type_t::f8e5m2);
    } else if (type == "f4e2m1" || type == "F4E2M1") {
        return ::ov::element::Type(::ov::element::Type_t::f4e2m1);
    } else if (type == "f8e8m0" || type == "F8E8M0") {
        return ::ov::element::Type(::ov::element::Type_t::f8e8m0);
    } else {
        OPENVINO_THROW("Incorrect type: ", type);
    }
}
}  // namespace

std::vector<const ov::element::Type*> ov::element::Type::get_known_types() {
    std::vector<const ov::element::Type*> rc = {
        &ov::element::dynamic, &ov::element::boolean, &ov::element::bf16,   &ov::element::f16,    &ov::element::f32,
        &ov::element::f64,     &ov::element::i4,      &ov::element::i8,     &ov::element::i16,    &ov::element::i32,
        &ov::element::i64,     &ov::element::u1,      &ov::element::u2,     &ov::element::u3,     &ov::element::u4,
        &ov::element::u6,      &ov::element::u8,      &ov::element::u16,    &ov::element::u32,    &ov::element::u64,
        &ov::element::nf4,     &ov::element::f8e4m3,  &ov::element::f8e5m2, &ov::element::string, &ov::element::f4e2m1,
        &ov::element::f8e8m0};
    return rc;
}

ov::element::Type::Type(size_t bitwidth,
                        bool is_real,
                        bool is_signed,
                        bool is_quantized,
                        const std::string& /* cname */) {
    const ElementTypes::ElementsMap elements_map{
        {ov::element::Type_t::undefined,
         {std::numeric_limits<size_t>::max(), false, false, false, "undefined", "undefined"}},
        {ov::element::Type_t::dynamic, {0, false, false, false, "dynamic", "dynamic"}},
        {ov::element::Type_t::boolean, {8, false, true, false, "char", "boolean"}},
        {ov::element::Type_t::bf16, {16, true, true, false, "bfloat16", "bf16"}},
        {ov::element::Type_t::f16, {16, true, true, false, "float16", "f16"}},
        {ov::element::Type_t::f32, {32, true, true, false, "float", "f32"}},
        {ov::element::Type_t::f64, {64, true, true, false, "double", "f64"}},
        {ov::element::Type_t::i4, {4, false, true, true, "int4_t", "i4"}},
        {ov::element::Type_t::i8, {8, false, true, true, "int8_t", "i8"}},
        {ov::element::Type_t::i16, {16, false, true, false, "int16_t", "i16"}},
        {ov::element::Type_t::i32, {32, false, true, true, "int32_t", "i32"}},
        {ov::element::Type_t::i64, {64, false, true, false, "int64_t", "i64"}},
        {ov::element::Type_t::u1, {1, false, false, false, "uint1_t", "u1"}},
        {ov::element::Type_t::u2, {2, false, false, false, "uint2_t", "u2"}},
        {ov::element::Type_t::u3, {3, false, false, false, "uint3_t", "u3"}},
        {ov::element::Type_t::u4, {4, false, false, false, "uint4_t", "u4"}},
        {ov::element::Type_t::u6, {6, false, false, false, "uint6_t", "u6"}},
        {ov::element::Type_t::u8, {8, false, false, true, "uint8_t", "u8"}},
        {ov::element::Type_t::u16, {16, false, false, false, "uint16_t", "u16"}},
        {ov::element::Type_t::u32, {32, false, false, false, "uint32_t", "u32"}},
        {ov::element::Type_t::u64, {64, false, false, false, "uint64_t", "u64"}},
        {ov::element::Type_t::nf4, {4, false, false, true, "nfloat4", "nf4"}},
        {ov::element::Type_t::f8e4m3, {8, true, true, true, "f8e4m3", "f8e4m3"}},
        {ov::element::Type_t::f8e5m2, {8, true, true, true, "f8e5m2", "f8e5m2"}},
        {ov::element::Type_t::string, {8 * sizeof(std::string), false, false, false, "string", "string"}},
        {ov::element::Type_t::f4e2m1, {4, true, true, true, "f4e2m1", "f4e2m1"}},
        {ov::element::Type_t::f8e8m0, {4, true, true, true, "f8e8m0", "f8e8m0"}},
    };
    for (const auto& t : elements_map) {
        const TypeInfo& info = t.second;
        if (bitwidth == info.m_bitwidth && is_real == info.m_is_real && is_signed == info.m_is_signed &&
            is_quantized == info.m_is_quantized) {
            m_type = t.first;
            return;
        }
    }
}

ov::element::Type::Type(const std::string& type) : Type(type_from_string(type)) {}

std::string ov::element::Type::c_type_string() const {
    return get_type_info(m_type).m_cname;
}

size_t ov::element::Type::size() const {
    return (bitwidth() + 7) >> 3;
}

size_t ov::element::Type::hash() const {
    return static_cast<size_t>(m_type);
}

std::string ov::element::Type::get_type_name() const {
    return to_string();
}

std::string ov::element::Type::to_string() const {
    return get_type_info(m_type).m_type_name;
}

namespace ov {
namespace element {
template <>
Type from<char>() {
    return Type_t::boolean;
}
template <>
Type from<bool>() {
    return Type_t::boolean;
}
template <>
Type from<ov::float16>() {
    return Type_t::f16;
}
template <>
Type from<float>() {
    return Type_t::f32;
}
template <>
Type from<double>() {
    return Type_t::f64;
}
template <>
Type from<int8_t>() {
    return Type_t::i8;
}
template <>
Type from<int16_t>() {
    return Type_t::i16;
}
template <>
Type from<int32_t>() {
    return Type_t::i32;
}
template <>
Type from<int64_t>() {
    return Type_t::i64;
}
template <>
Type from<uint8_t>() {
    return Type_t::u8;
}
template <>
Type from<uint16_t>() {
    return Type_t::u16;
}
template <>
Type from<uint32_t>() {
    return Type_t::u32;
}
template <>
Type from<uint64_t>() {
    return Type_t::u64;
}
template <>
Type from<ov::bfloat16>() {
    return Type_t::bf16;
}
template <>
Type from<ov::float8_e4m3>() {
    return Type_t::f8e4m3;
}
template <>
Type from<ov::float8_e5m2>() {
    return Type_t::f8e5m2;
}
template <>
Type from<std::string>() {
    return Type_t::string;
}
template <>
Type from<ov::float4_e2m1>() {
    return Type_t::f4e2m1;
}
template <>
Type from<ov::float8_e8m0>() {
    return Type_t::f8e8m0;
}

Type fundamental_type_for(const Type& type) {
    switch (type) {
    case Type_t::boolean:
        return from<element_type_traits<Type_t::boolean>::value_type>();
    case Type_t::bf16:
        return from<element_type_traits<Type_t::bf16>::value_type>();
    case Type_t::f16:
        return from<element_type_traits<Type_t::f16>::value_type>();
    case Type_t::f32:
        return from<element_type_traits<Type_t::f32>::value_type>();
    case Type_t::f64:
        return from<element_type_traits<Type_t::f64>::value_type>();
    case Type_t::f8e4m3:
        return from<element_type_traits<Type_t::f8e4m3>::value_type>();
    case Type_t::f8e5m2:
        return from<element_type_traits<Type_t::f8e5m2>::value_type>();
    case Type_t::i4:
        return from<element_type_traits<Type_t::i4>::value_type>();
    case Type_t::i8:
        return from<element_type_traits<Type_t::i8>::value_type>();
    case Type_t::i16:
        return from<element_type_traits<Type_t::i16>::value_type>();
    case Type_t::i32:
        return from<element_type_traits<Type_t::i32>::value_type>();
    case Type_t::i64:
        return from<element_type_traits<Type_t::i64>::value_type>();
    case Type_t::u1:
        return from<element_type_traits<Type_t::u1>::value_type>();
    case Type_t::u2:
        return from<element_type_traits<Type_t::u2>::value_type>();
    case Type_t::u3:
        return from<element_type_traits<Type_t::u3>::value_type>();
    case Type_t::u4:
        return from<element_type_traits<Type_t::u4>::value_type>();
    case Type_t::u6:
        return from<element_type_traits<Type_t::u6>::value_type>();
    case Type_t::u8:
        return from<element_type_traits<Type_t::u8>::value_type>();
    case Type_t::u16:
        return from<element_type_traits<Type_t::u16>::value_type>();
    case Type_t::u32:
        return from<element_type_traits<Type_t::u32>::value_type>();
    case Type_t::u64:
        return from<element_type_traits<Type_t::u64>::value_type>();
    case Type_t::nf4:
        return from<element_type_traits<Type_t::nf4>::value_type>();
    case Type_t::string:
        return from<element_type_traits<Type_t::string>::value_type>();
    case Type_t::f4e2m1:
        return from<element_type_traits<Type_t::f4e2m1>::value_type>();
    case Type_t::f8e8m0:
        return from<element_type_traits<Type_t::f8e8m0>::value_type>();
    default:
        OPENVINO_THROW("Unsupported Data type: ", type);
    }
}

}  // namespace element
}  // namespace ov

std::ostream& ov::element::operator<<(std::ostream& out, const ov::element::Type& obj) {
    return out << obj.to_string();
}

std::istream& ov::element::operator>>(std::istream& in, ov::element::Type& obj) {
    const std::unordered_map<std::string, ov::element::Type> legacy = {
        {"BOOL", ov::element::boolean},  {"BF16", ov::element::bf16},     {"I4", ov::element::i4},
        {"I8", ov::element::i8},         {"I16", ov::element::i16},       {"I32", ov::element::i32},
        {"I64", ov::element::i64},       {"U4", ov::element::u4},         {"U8", ov::element::u8},
        {"U16", ov::element::u16},       {"U32", ov::element::u32},       {"U64", ov::element::u64},
        {"FP32", ov::element::f32},      {"FP64", ov::element::f64},      {"FP16", ov::element::f16},
        {"BIN", ov::element::u1},        {"NF4", ov::element::nf4},       {"F8E4M3", ov::element::f8e4m3},
        {"F8E5M2", ov::element::f8e5m2}, {"STRING", ov::element::string}, {"F4E2M1", ov::element::f4e2m1},
        {"F8E8M0", ov::element::f8e8m0}};
    std::string str;
    in >> str;
    auto it_legacy = legacy.find(str);
    if (it_legacy != legacy.end()) {
        obj = it_legacy->second;
        return in;
    }
    for (auto&& type : Type::get_known_types()) {
        if (type->to_string() == str) {
            obj = *type;
            break;
        }
    }
    return in;
}

bool ov::element::Type::compatible(const ov::element::Type& t) const {
    return (is_dynamic() || t.is_dynamic() || *this == t);
}

bool ov::element::Type::merge(ov::element::Type& dst, const ov::element::Type& t1, const ov::element::Type& t2) {
    if (t1.is_dynamic()) {
        dst = t2;
        return true;
    } else if (t2.is_dynamic()) {
        dst = t1;
        return true;
    } else if (t1 == t2) {
        dst = t1;
        return true;
    } else {
        return false;
    }
}

bool ov::element::Type::is_static() const {
    return get_type_info(m_type).m_bitwidth != 0;
}

bool ov::element::Type::is_real() const {
    return get_type_info(m_type).m_is_real;
}

bool ov::element::Type::is_integral_number() const {
    return is_integral() && (m_type != ov::element::boolean);
}

bool ov::element::Type::is_signed() const {
    return get_type_info(m_type).m_is_signed;
}

bool ov::element::Type::is_quantized() const {
    return get_type_info(m_type).m_is_quantized;
}

size_t ov::element::Type::bitwidth() const {
    return get_type_info(m_type).m_bitwidth;
}

inline size_t compiler_byte_size(ov::element::Type_t et) {
    switch (et) {
#define ET_CASE(et)               \
    case ov::element::Type_t::et: \
        return sizeof(ov::element_type_traits<ov::element::Type_t::et>::value_type);
        ET_CASE(boolean);
        ET_CASE(bf16);
        ET_CASE(f16);
        ET_CASE(f32);
        ET_CASE(f64);
        ET_CASE(i4);
        ET_CASE(i8);
        ET_CASE(i16);
        ET_CASE(i32);
        ET_CASE(i64);
        ET_CASE(u1);
        ET_CASE(u2);
        ET_CASE(u3);
        ET_CASE(u4);
        ET_CASE(u6);
        ET_CASE(u8);
        ET_CASE(u16);
        ET_CASE(u32);
        ET_CASE(u64);
        ET_CASE(nf4);
        ET_CASE(f8e4m3);
        ET_CASE(f8e5m2);
        ET_CASE(string);
        ET_CASE(f4e2m1);
        ET_CASE(f8e8m0);
#undef ET_CASE
    case ov::element::Type_t::undefined:
        return 0;
    case ov::element::Type_t::dynamic:
        return 0;
    }

    OPENVINO_THROW("compiler_byte_size: Unsupported value of ov::element::Type_t: ", static_cast<int>(et));
}

namespace ov {
template <>
OPENVINO_API EnumNames<element::Type_t>& EnumNames<element::Type_t>::get() {
    static auto enum_names = EnumNames<element::Type_t>("element::Type_t",
                                                        {{"undefined", element::Type_t::undefined},
                                                         {"dynamic", element::Type_t::dynamic},
                                                         {"boolean", element::Type_t::boolean},
                                                         {"bf16", element::Type_t::bf16},
                                                         {"f16", element::Type_t::f16},
                                                         {"f32", element::Type_t::f32},
                                                         {"f64", element::Type_t::f64},
                                                         {"i4", element::Type_t::i4},
                                                         {"i8", element::Type_t::i8},
                                                         {"i16", element::Type_t::i16},
                                                         {"i32", element::Type_t::i32},
                                                         {"i64", element::Type_t::i64},
                                                         {"u1", element::Type_t::u1},
                                                         {"u2", element::Type_t::u2},
                                                         {"u3", element::Type_t::u3},
                                                         {"u4", element::Type_t::u4},
                                                         {"u6", element::Type_t::u6},
                                                         {"u8", element::Type_t::u8},
                                                         {"u16", element::Type_t::u16},
                                                         {"u32", element::Type_t::u32},
                                                         {"u64", element::Type_t::u64},
                                                         {"nf4", element::Type_t::nf4},
                                                         {"f8e4m3", element::Type_t::f8e4m3},
                                                         {"f8e5m2", element::Type_t::f8e5m2},
                                                         {"string", element::Type_t::string},
                                                         {"f4e2m1", element::Type_t::f4e2m1},
                                                         {"f8e8m0", element::Type_t::f8e8m0}});
    return enum_names;
}

const std::string& AttributeAdapter<element::Type>::get() {
    return as_string(static_cast<element::Type_t>(m_ref));
}

void AttributeAdapter<element::Type>::set(const std::string& value) {
    m_ref = as_enum<element::Type_t>(value);
}
}  // namespace ov
