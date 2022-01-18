// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/element_type.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <unordered_map>

#include "ngraph/log.hpp"
#include "ngraph/type/element_type_traits.hpp"

BWDCMP_RTTI_DEFINITION(ov::AttributeAdapter<ov::element::Type>);
BWDCMP_RTTI_DEFINITION(ov::AttributeAdapter<ov::element::TypeVector>);

namespace {
class TypeInfo {
public:
    TypeInfo(size_t bitwidth,
             bool is_real,
             bool is_signed,
             bool is_quantized,
             const std::string& cname,
             const std::string& type_name)
        : m_bitwidth{bitwidth},
          m_is_real{is_real},
          m_is_signed{is_signed},
          m_is_quantized{is_quantized},
          m_cname{cname},
          m_type_name{type_name} {}
    size_t m_bitwidth;
    bool m_is_real;
    bool m_is_signed;
    bool m_is_quantized;
    std::string m_cname;
    std::string m_type_name;
};

struct ElementTypes {
    struct TypeHash {
        size_t operator()(ov::element::Type_t t) const {
            return static_cast<size_t>(t);
        }
    };

    using ElementsMap = std::unordered_map<ov::element::Type_t, TypeInfo, TypeHash>;
};

const ElementTypes::ElementsMap& get_type_info_map() {
    static const ElementTypes::ElementsMap elements_map{
        {ov::element::Type_t::undefined,
         TypeInfo(std::numeric_limits<size_t>::max(), false, false, false, "undefined", "undefined")},
        {ov::element::Type_t::dynamic, TypeInfo(0, false, false, false, "dynamic", "dynamic")},
        {ov::element::Type_t::boolean, TypeInfo(8, false, true, false, "char", "boolean")},
        {ov::element::Type_t::bf16, TypeInfo(16, true, true, false, "bfloat16", "bf16")},
        {ov::element::Type_t::f16, TypeInfo(16, true, true, false, "float16", "f16")},
        {ov::element::Type_t::f32, TypeInfo(32, true, true, false, "float", "f32")},
        {ov::element::Type_t::f64, TypeInfo(64, true, true, false, "double", "f64")},
        {ov::element::Type_t::i4, TypeInfo(4, false, true, true, "int4_t", "i4")},
        {ov::element::Type_t::i8, TypeInfo(8, false, true, true, "int8_t", "i8")},
        {ov::element::Type_t::i16, TypeInfo(16, false, true, false, "int16_t", "i16")},
        {ov::element::Type_t::i32, TypeInfo(32, false, true, true, "int32_t", "i32")},
        {ov::element::Type_t::i64, TypeInfo(64, false, true, false, "int64_t", "i64")},
        {ov::element::Type_t::u1, TypeInfo(1, false, false, false, "uint1_t", "u1")},
        {ov::element::Type_t::u4, TypeInfo(4, false, false, false, "uint4_t", "u4")},
        {ov::element::Type_t::u8, TypeInfo(8, false, false, true, "uint8_t", "u8")},
        {ov::element::Type_t::u16, TypeInfo(16, false, false, false, "uint16_t", "u16")},
        {ov::element::Type_t::u32, TypeInfo(32, false, false, false, "uint32_t", "u32")},
        {ov::element::Type_t::u64, TypeInfo(64, false, false, false, "uint64_t", "u64")},
    };
    return elements_map;
};

const TypeInfo& get_type_info(ov::element::Type_t type) {
    const auto& tim = get_type_info_map();
    const auto& found = tim.find(type);
    if (found == tim.end()) {
        throw std::out_of_range{"ov::element::Type_t not supported"};
    }
    return found->second;
};
}  // namespace

std::vector<const ov::element::Type*> ov::element::Type::get_known_types() {
    std::vector<const ov::element::Type*> rc = {&ov::element::dynamic,
                                                &ov::element::boolean,
                                                &ov::element::bf16,
                                                &ov::element::f16,
                                                &ov::element::f32,
                                                &ov::element::f64,
                                                &ov::element::i4,
                                                &ov::element::i8,
                                                &ov::element::i16,
                                                &ov::element::i32,
                                                &ov::element::i64,
                                                &ov::element::u1,
                                                &ov::element::u4,
                                                &ov::element::u8,
                                                &ov::element::u16,
                                                &ov::element::u32,
                                                &ov::element::u64};
    return rc;
}

ov::element::Type::Type(size_t bitwidth,
                        bool is_real,
                        bool is_signed,
                        bool is_quantized,
                        const std::string& /* cname */) {
    for (const auto& t : get_type_info_map()) {
        const TypeInfo& info = t.second;
        if (bitwidth == info.m_bitwidth && is_real == info.m_is_real && is_signed == info.m_is_signed &&
            is_quantized == info.m_is_quantized) {
            m_type = t.first;
            return;
        }
    }
}

const std::string& ov::element::Type::c_type_string() const {
    return get_type_info(m_type).m_cname;
}

size_t ov::element::Type::size() const {
    return std::ceil(static_cast<float>(bitwidth()) / 8.0f);
}

size_t ov::element::Type::hash() const {
    return static_cast<size_t>(m_type);
}

const std::string& ov::element::Type::get_type_name() const {
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
    case Type_t::u4:
        return from<element_type_traits<Type_t::u4>::value_type>();
    case Type_t::u8:
        return from<element_type_traits<Type_t::u8>::value_type>();
    case Type_t::u16:
        return from<element_type_traits<Type_t::u16>::value_type>();
    case Type_t::u32:
        return from<element_type_traits<Type_t::u32>::value_type>();
    case Type_t::u64:
        return from<element_type_traits<Type_t::u64>::value_type>();
    default:
        OPENVINO_UNREACHABLE("Unsupported Data type: ", type);
    }
}

}  // namespace element
}  // namespace ov

std::ostream& ov::element::operator<<(std::ostream& out, const ov::element::Type& obj) {
    return out << obj.get_type_name();
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
        ET_CASE(u4);
        ET_CASE(u8);
        ET_CASE(u16);
        ET_CASE(u32);
        ET_CASE(u64);
#undef ET_CASE
    case ov::element::Type_t::undefined:
        return 0;
    case ov::element::Type_t::dynamic:
        return 0;
    }

    throw ov::Exception("compiler_byte_size: Unsupported value of ov::element::Type_t: " +
                        std::to_string(static_cast<int>(et)));
}

namespace ov {
template <>
NGRAPH_API EnumNames<element::Type_t>& EnumNames<element::Type_t>::get() {
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
                                                         {"u4", element::Type_t::u4},
                                                         {"u8", element::Type_t::u8},
                                                         {"u16", element::Type_t::u16},
                                                         {"u32", element::Type_t::u32},
                                                         {"u64", element::Type_t::u64}});
    return enum_names;
}

BWDCMP_RTTI_DEFINITION(AttributeAdapter<element::Type_t>);

const std::string& AttributeAdapter<element::Type>::get() {
    return as_string(static_cast<element::Type_t>(m_ref));
}

void AttributeAdapter<element::Type>::set(const std::string& value) {
    m_ref = as_enum<element::Type_t>(value);
}
}  // namespace ov
