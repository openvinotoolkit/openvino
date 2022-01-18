// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//================================================================================================
// ElementType
//================================================================================================

#pragma once

#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/core/core_visibility.hpp"
#include "openvino/core/deprecated.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov {
namespace element {
enum class Type_t {
    undefined,
    dynamic,
    boolean,
    bf16,
    f16,
    f32,
    f64,
    i4,
    i8,
    i16,
    i32,
    i64,
    u1,
    u4,
    u8,
    u16,
    u32,
    u64
};

class OPENVINO_API Type {
public:
    Type() = default;
    Type(const Type&) = default;
    constexpr Type(const Type_t t) : m_type{t} {}
    Type(size_t bitwidth, bool is_real, bool is_signed, bool is_quantized, const std::string& cname);
    Type& operator=(const Type&) = default;
    const std::string& c_type_string() const;
    size_t size() const;
    size_t hash() const;
    bool is_static() const;
    bool is_dynamic() const {
        return !is_static();
    }
    bool is_real() const;
    // TODO: We may want to revisit this definition when we do a more general cleanup of
    // element types:
    bool is_integral() const {
        return !is_real();
    }
    bool is_integral_number() const;
    bool is_signed() const;
    bool is_quantized() const;
    size_t bitwidth() const;
    // The name of this type, the enum name of this type
    const std::string& get_type_name() const;
    friend OPENVINO_API std::ostream& operator<<(std::ostream&, const Type&);
    static std::vector<const Type*> get_known_types();

    /// \brief Checks whether this element type is merge-compatible with `t`.
    /// \param t The element type to compare this element type to.
    /// \return `true` if this element type is compatible with `t`, else `false`.
    bool compatible(const element::Type& t) const;

    /// \brief Merges two element types t1 and t2, writing the result into dst and
    ///        returning true if successful, else returning false.
    ///
    ///        To "merge" two element types t1 and t2 is to find the least restrictive
    ///        element type t that is no more restrictive than t1 and t2, if t exists.
    ///        More simply:
    ///
    ///           merge(dst,element::Type::dynamic,t)
    ///              writes t to dst and returns true
    ///
    ///           merge(dst,t,element::Type::dynamic)
    ///              writes t to dst and returns true
    ///
    ///           merge(dst,t1,t2) where t1, t2 both static and equal
    ///              writes t1 to dst and returns true
    ///
    ///           merge(dst,t1,t2) where t1, t2 both static and unequal
    ///              does nothing to dst, and returns false
    static bool merge(element::Type& dst, const element::Type& t1, const element::Type& t2);

    // \brief This allows switch(element_type)
    constexpr operator Type_t() const {
        return m_type;
    }

private:
    Type_t m_type{Type_t::undefined};
};

using TypeVector = std::vector<Type>;

constexpr Type undefined(Type_t::undefined);
constexpr Type dynamic(Type_t::dynamic);
constexpr Type boolean(Type_t::boolean);
constexpr Type bf16(Type_t::bf16);
constexpr Type f16(Type_t::f16);
constexpr Type f32(Type_t::f32);
constexpr Type f64(Type_t::f64);
constexpr Type i4(Type_t::i4);
constexpr Type i8(Type_t::i8);
constexpr Type i16(Type_t::i16);
constexpr Type i32(Type_t::i32);
constexpr Type i64(Type_t::i64);
constexpr Type u1(Type_t::u1);
constexpr Type u4(Type_t::u4);
constexpr Type u8(Type_t::u8);
constexpr Type u16(Type_t::u16);
constexpr Type u32(Type_t::u32);
constexpr Type u64(Type_t::u64);

template <typename T>
Type from() {
    throw std::invalid_argument("Unknown type");
}
template <>
OPENVINO_API Type from<char>();
template <>
OPENVINO_API Type from<bool>();
template <>
OPENVINO_API Type from<float>();
template <>
OPENVINO_API Type from<double>();
template <>
OPENVINO_API Type from<int8_t>();
template <>
OPENVINO_API Type from<int16_t>();
template <>
OPENVINO_API Type from<int32_t>();
template <>
OPENVINO_API Type from<int64_t>();
template <>
OPENVINO_API Type from<uint8_t>();
template <>
OPENVINO_API Type from<uint16_t>();
template <>
OPENVINO_API Type from<uint32_t>();
template <>
OPENVINO_API Type from<uint64_t>();
template <>
OPENVINO_API Type from<ov::bfloat16>();
template <>
OPENVINO_API Type from<ov::float16>();

OPENVINO_API Type fundamental_type_for(const Type& type);

OPENVINO_API
std::ostream& operator<<(std::ostream& out, const ov::element::Type& obj);
}  // namespace element

template <>
class OPENVINO_API AttributeAdapter<ov::element::Type_t> : public EnumAttributeAdapterBase<ov::element::Type_t> {
public:
    AttributeAdapter(ov::element::Type_t& value) : EnumAttributeAdapterBase<ov::element::Type_t>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::element::Type_t>");
    BWDCMP_RTTI_DECLARATION;
};

template <>
class OPENVINO_API AttributeAdapter<ov::element::Type> : public ValueAccessor<std::string> {
public:
    OPENVINO_RTTI("AttributeAdapter<ov::element::Type>");
    BWDCMP_RTTI_DECLARATION;
    AttributeAdapter(ov::element::Type& value) : m_ref(value) {}

    const std::string& get() override;
    void set(const std::string& value) override;

    operator ov::element::Type&() {
        return m_ref;
    }

protected:
    ov::element::Type& m_ref;
};

template <>
class OPENVINO_API AttributeAdapter<ov::element::TypeVector> : public DirectValueAccessor<ov::element::TypeVector> {
public:
    OPENVINO_RTTI("AttributeAdapter<ov::element::TypeVector>");
    BWDCMP_RTTI_DECLARATION;
    AttributeAdapter(ov::element::TypeVector& value) : DirectValueAccessor<ov::element::TypeVector>(value) {}
};

}  // namespace ov
