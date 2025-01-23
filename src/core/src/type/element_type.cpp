// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/element_type.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <unordered_map>

#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/util/common_util.hpp"

namespace ov::element {
namespace {
constexpr size_t idx(Type_t e) noexcept {
    return static_cast<std::underlying_type_t<Type_t>>(e);
}

// Update it when new type is added
constexpr size_t enum_types_size = idx(f8e8m0) + 1;

struct TypeInfo {
    size_t m_bitwidth;
    bool m_is_real;
    bool m_is_signed;
    bool m_is_quantized;
    const char* m_cname;
    const char* m_type_name;
    const std::initializer_list<const char*> m_aliases;

    bool has_name(const std::string& type) const {
        return type == m_type_name || std::find(m_aliases.begin(), m_aliases.end(), type) != m_aliases.end();
    }

    constexpr bool is_valid() const {
        constexpr auto null_info = TypeInfo{};
        return std::tie(m_cname, m_type_name) == std::tie(null_info.m_cname, null_info.m_type_name);
    }
};

static constexpr std::array<TypeInfo, enum_types_size> types_info{
    TypeInfo{std::numeric_limits<size_t>::max(),
             false,
             false,
             false,
             "undefined",
             "undefined",
             {"UNSPECIFIED"}},                                                       // undefined
    {0, false, false, false, "dynamic", "dynamic", {}},                              // dynamic
    {8, false, true, false, "char", "boolean", {"BOOL"}},                            // boolean
    {16, true, true, false, "bfloat16", "bf16", {"BF16"}},                           // bf16
    {16, true, true, false, "float16", "f16", {"FP16"}},                             // f16
    {32, true, true, false, "float", "f32", {"FP32"}},                               // f32
    {64, true, true, false, "double", "f64", {"FP64"}},                              // f64
    {4, false, true, true, "int4_t", "i4", {"I4"}},                                  // i4
    {8, false, true, true, "int8_t", "i8", {"I8"}},                                  // i8
    {16, false, true, false, "int16_t", "i16", {"I16"}},                             // i16
    {32, false, true, true, "int32_t", "i32", {"I32"}},                              // i32
    {64, false, true, false, "int64_t", "i64", {"I64"}},                             // i64
    {1, false, false, false, "uint1_t", "u1", {"U1", "bin", "BIN"}},                 // u1
    {2, false, false, false, "uint2_t", "u2", {"U2"}},                               // u2
    {3, false, false, false, "uint3_t", "u3", {"U3"}},                               // u3
    {4, false, false, false, "uint4_t", "u4", {"U4"}},                               // u4
    {6, false, false, false, "uint6_t", "u6", {"U6"}},                               // u6
    {8, false, false, true, "uint8_t", "u8", {"U8"}},                                // u8
    {16, false, false, false, "uint16_t", "u16", {"U16"}},                           // u16
    {32, false, false, false, "uint32_t", "u32", {"U32"}},                           // u32
    {64, false, false, false, "uint64_t", "u64", {"U64"}},                           // u64
    {4, false, false, true, "nfloat4", "nf4", {"NF4"}},                              // nf4
    {8, true, true, true, "f8e4m3", "f8e4m3", {"F8E4M3"}},                           // f8e4m3
    {8, true, true, true, "f8e5m2", "f8e5m2", {"F8E5M2"}},                           // f8e5m2
    {8 * sizeof(std::string), false, false, false, "string", "string", {"STRING"}},  // string
    {4, true, true, true, "f4e2m1", "f4e2m1", {"F4E2M1"}},                           // f4e2m1
    {8, true, true, true, "f8e8m0", "f8e8m0", {"F8E8M0"}}                            // f8e8m0
};

constexpr bool validate_types_info(decltype(types_info)& info, size_t i = 0) {
    return i >= info.size() ? true : info[i].is_valid() ? false : validate_types_info(info, i + 1);
}

static_assert(validate_types_info(types_info), "Some entries of type_info  have not valid information");

constexpr bool is_valid_type_idx(size_t idx) {
    return idx < types_info.size();
}

size_t type_idx_for(const std::string& type_name) {
    size_t type_idx = 0;
    for (; is_valid_type_idx(type_idx); ++type_idx) {
        if (types_info[type_idx].has_name(type_name)) {
            break;
        }
    }
    return type_idx;
}

const TypeInfo& get_type_info(Type_t type) {
    const auto type_idx = idx(type);
    OPENVINO_ASSERT(is_valid_type_idx(type_idx), "Type_t not supported: ", type_idx);
    return types_info[type_idx];
}

Type type_from_string(const std::string& type) {
    const auto type_idx = type_idx_for(type);
    OPENVINO_ASSERT(is_valid_type_idx(type_idx), "Unsupported element type: ", type);
    return {static_cast<Type_t>(type_idx)};
}
}  // namespace

std::vector<const Type*> Type::get_known_types() {
    // clang-format off
    static constexpr auto known_types = ov::util::make_array(
        &dynamic, &boolean, &bf16, &f16,    &f32,    &f64,    &i4,     &i8,    &i16,
        &i32,     &i64,     &u1,   &u2,     &u3,     &u4,     &u6,     &u8,    &u16,
        &u32,     &u64,     &nf4,  &f8e4m3, &f8e5m2, &string, &f4e2m1, &f8e8m0);
    // clang-format on
    return {known_types.begin(), known_types.end()};
}

Type::Type(const std::string& type) : Type(type_from_string(type)) {}

std::string Type::c_type_string() const {
    return get_type_info(m_type).m_cname;
}

size_t Type::size() const {
    return (bitwidth() + 7) >> 3;
}

size_t Type::hash() const {
    return static_cast<size_t>(m_type);
}

std::string Type::get_type_name() const {
    return to_string();
}

std::string Type::to_string() const {
    return get_type_info(m_type).m_type_name;
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

std::ostream& operator<<(std::ostream& out, const Type& obj) {
    return out << obj.to_string();
}

std::istream& operator>>(std::istream& in, Type& obj) {
    std::string str;
    in >> str;
    if (const auto type_idx = type_idx_for(str); is_valid_type_idx(type_idx)) {
        obj = {static_cast<Type_t>(type_idx)};
    }
    return in;
}

bool Type::compatible(const Type& t) const {
    return (is_dynamic() || t.is_dynamic() || *this == t);
}

bool Type::merge(Type& dst, const Type& t1, const Type& t2) {
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

bool Type::is_static() const {
    return get_type_info(m_type).m_bitwidth != 0;
}

bool Type::is_real() const {
    return get_type_info(m_type).m_is_real;
}

bool Type::is_integral_number() const {
    return is_integral() && (m_type != boolean);
}

bool Type::is_signed() const {
    return get_type_info(m_type).m_is_signed;
}

bool Type::is_quantized() const {
    return get_type_info(m_type).m_is_quantized;
}

size_t Type::bitwidth() const {
    return get_type_info(m_type).m_bitwidth;
}
}  // namespace ov::element

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
