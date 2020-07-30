//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <cmath>
#include <iostream>
#include <map>

#include "ngraph/log.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/type/element_type_traits.hpp"

using namespace ngraph;
using namespace std;

const element::Type element::undefined(element::Type_t::undefined);
const element::Type element::dynamic(element::Type_t::dynamic);
const element::Type element::boolean(element::Type_t::boolean);
const element::Type element::bf16(element::Type_t::bf16);
const element::Type element::f16(element::Type_t::f16);
const element::Type element::f32(element::Type_t::f32);
const element::Type element::f64(element::Type_t::f64);
const element::Type element::i8(element::Type_t::i8);
const element::Type element::i16(element::Type_t::i16);
const element::Type element::i32(element::Type_t::i32);
const element::Type element::i64(element::Type_t::i64);
const element::Type element::u1(element::Type_t::u1);
const element::Type element::u8(element::Type_t::u8);
const element::Type element::u16(element::Type_t::u16);
const element::Type element::u32(element::Type_t::u32);
const element::Type element::u64(element::Type_t::u64);

constexpr DiscreteTypeInfo AttributeAdapter<element::Type>::type_info;

class TypeInfo
{
public:
    TypeInfo(size_t bitwidth,
             bool is_real,
             bool is_signed,
             bool is_quantized,
             const std::string& cname,
             const std::string& type_name)
        : m_bitwidth{bitwidth}
        , m_is_real{is_real}
        , m_is_signed{is_signed}
        , m_is_quantized{is_quantized}
        , m_cname{cname}
        , m_type_name{type_name}
    {
    }
    size_t m_bitwidth;
    bool m_is_real;
    bool m_is_signed;
    bool m_is_quantized;
    std::string m_cname;
    std::string m_type_name;
};

static const map<element::Type_t, const TypeInfo>& get_type_info_map()
{
    static map<element::Type_t, const TypeInfo> s_type_info_map{
        {element::Type_t::undefined,
         TypeInfo(
             std::numeric_limits<size_t>::max(), false, false, false, "undefined", "undefined")},
        {element::Type_t::dynamic, TypeInfo(0, false, false, false, "dynamic", "dynamic")},
        {element::Type_t::boolean, TypeInfo(8, false, true, false, "char", "boolean")},
        {element::Type_t::bf16, TypeInfo(16, true, true, false, "bfloat16", "bf16")},
        {element::Type_t::f16, TypeInfo(16, true, true, false, "float16", "f16")},
        {element::Type_t::f32, TypeInfo(32, true, true, false, "float", "f32")},
        {element::Type_t::f64, TypeInfo(64, true, true, false, "double", "f64")},
        {element::Type_t::i8, TypeInfo(8, false, true, true, "int8_t", "i8")},
        {element::Type_t::i16, TypeInfo(16, false, true, false, "int16_t", "i16")},
        {element::Type_t::i32, TypeInfo(32, false, true, true, "int32_t", "i32")},
        {element::Type_t::i64, TypeInfo(64, false, true, false, "int64_t", "i64")},
        {element::Type_t::u1, TypeInfo(1, false, false, false, "uint1_t", "u1")},
        {element::Type_t::u8, TypeInfo(8, false, false, true, "uint8_t", "u8")},
        {element::Type_t::u16, TypeInfo(16, false, false, false, "uint16_t", "u16")},
        {element::Type_t::u32, TypeInfo(32, false, false, false, "uint32_t", "u32")},
        {element::Type_t::u64, TypeInfo(64, false, false, false, "uint64_t", "u64")},
    };
    return s_type_info_map;
};

std::vector<const element::Type*> element::Type::get_known_types()
{
    std::vector<const element::Type*> rc = {&element::dynamic,
                                            &element::boolean,
                                            &element::bf16,
                                            &element::f16,
                                            &element::f32,
                                            &element::f64,
                                            &element::i8,
                                            &element::i16,
                                            &element::i32,
                                            &element::i64,
                                            &element::u1,
                                            &element::u8,
                                            &element::u16,
                                            &element::u32,
                                            &element::u64};
    return rc;
}

element::Type::Type(size_t bitwidth,
                    bool is_real,
                    bool is_signed,
                    bool is_quantized,
                    const std::string& /* cname */)
{
    for (auto& t : get_type_info_map())
    {
        const TypeInfo& info = t.second;
        if (bitwidth == info.m_bitwidth && is_real == info.m_is_real &&
            is_signed == info.m_is_signed && is_quantized == info.m_is_quantized)
        {
            m_type = t.first;
            return;
        }
    }
}

const std::string& element::Type::c_type_string() const
{
    return get_type_info_map().at(m_type).m_cname;
}

bool element::Type::operator==(const element::Type& other) const
{
    return m_type == other.m_type;
}

bool element::Type::operator<(const Type& other) const
{
    return m_type < other.m_type;
}

size_t element::Type::size() const
{
    return std::ceil(static_cast<float>(bitwidth()) / 8.0f);
}

size_t element::Type::hash() const
{
    return static_cast<size_t>(m_type);
}

const std::string& element::Type::get_type_name() const
{
    return get_type_info_map().at(m_type).m_type_name;
}

namespace ngraph
{
    namespace element
    {
        template <>
        Type from<char>()
        {
            return Type_t::boolean;
        }
        template <>
        Type from<bool>()
        {
            return Type_t::boolean;
        }
        template <>
        Type from<ngraph::float16>()
        {
            return Type_t::f16;
        }
        template <>
        Type from<float>()
        {
            return Type_t::f32;
        }
        template <>
        Type from<double>()
        {
            return Type_t::f64;
        }
        template <>
        Type from<int8_t>()
        {
            return Type_t::i8;
        }
        template <>
        Type from<int16_t>()
        {
            return Type_t::i16;
        }
        template <>
        Type from<int32_t>()
        {
            return Type_t::i32;
        }
        template <>
        Type from<int64_t>()
        {
            return Type_t::i64;
        }
        template <>
        Type from<uint8_t>()
        {
            return Type_t::u8;
        }
        template <>
        Type from<uint16_t>()
        {
            return Type_t::u16;
        }
        template <>
        Type from<uint32_t>()
        {
            return Type_t::u32;
        }
        template <>
        Type from<uint64_t>()
        {
            return Type_t::u64;
        }
        template <>
        Type from<ngraph::bfloat16>()
        {
            return Type_t::bf16;
        }
    }
}

std::ostream& element::operator<<(std::ostream& out, const element::Type& obj)
{
    return out << obj.get_type_name();
}

bool element::Type::compatible(const element::Type& t) const
{
    return (is_dynamic() || t.is_dynamic() || *this == t);
}

bool element::Type::merge(element::Type& dst, const element::Type& t1, const element::Type& t2)
{
    if (t1.is_dynamic())
    {
        dst = t2;
        return true;
    }
    else if (t2.is_dynamic())
    {
        dst = t1;
        return true;
    }
    else if (t1 == t2)
    {
        dst = t1;
        return true;
    }
    else
    {
        return false;
    }
}

bool element::Type::is_static() const
{
    return get_type_info_map().at(m_type).m_bitwidth != 0;
}

bool element::Type::is_real() const
{
    return get_type_info_map().at(m_type).m_is_real;
}

bool element::Type::is_integral_number() const
{
    return is_integral() && (m_type != element::boolean);
}

bool element::Type::is_signed() const
{
    return get_type_info_map().at(m_type).m_is_signed;
}

bool element::Type::is_quantized() const
{
    return get_type_info_map().at(m_type).m_is_quantized;
}

size_t element::Type::bitwidth() const
{
    return get_type_info_map().at(m_type).m_bitwidth;
}

size_t ngraph::compiler_byte_size(element::Type_t et)
{
    switch (et)
    {
#define ET_CASE(et)                                                                                \
    case element::Type_t::et: return sizeof(element_type_traits<element::Type_t::et>::value_type);
        ET_CASE(boolean);
        ET_CASE(bf16);
        ET_CASE(f16);
        ET_CASE(f32);
        ET_CASE(f64);
        ET_CASE(i8);
        ET_CASE(i16);
        ET_CASE(i32);
        ET_CASE(i64);
        ET_CASE(u1);
        ET_CASE(u8);
        ET_CASE(u16);
        ET_CASE(u32);
        ET_CASE(u64);
#undef ET_CASE
    case element::Type_t::undefined: return 0;
    case element::Type_t::dynamic: return 0;
    }

    throw ngraph_error("compiler_byte_size: Unsupported value of element::Type_t: " +
                       std::to_string(static_cast<int>(et)));
}

namespace ngraph
{
    template <>
    NGRAPH_API EnumNames<element::Type_t>& EnumNames<element::Type_t>::get()
    {
        static auto enum_names =
            EnumNames<element::Type_t>("element::Type_t",
                                       {{"undefined", element::Type_t::undefined},
                                        {"dynamic", element::Type_t::dynamic},
                                        {"boolean", element::Type_t::boolean},
                                        {"bf16", element::Type_t::bf16},
                                        {"f16", element::Type_t::f16},
                                        {"f32", element::Type_t::f32},
                                        {"f64", element::Type_t::f64},
                                        {"i8", element::Type_t::i8},
                                        {"i16", element::Type_t::i16},
                                        {"i32", element::Type_t::i32},
                                        {"i64", element::Type_t::i64},
                                        {"u8", element::Type_t::u8},
                                        {"u16", element::Type_t::u16},
                                        {"u32", element::Type_t::u32},
                                        {"u64", element::Type_t::u64}});
        return enum_names;
    }
}

constexpr DiscreteTypeInfo AttributeAdapter<element::Type_t>::type_info;

const std::string& AttributeAdapter<element::Type>::get()
{
    return as_string(static_cast<element::Type_t>(m_ref));
}

void AttributeAdapter<element::Type>::set(const std::string& value)
{
    m_ref = as_enum<element::Type_t>(value);
}
