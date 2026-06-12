// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/element_type.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <string_view>
#include <unordered_map>

#include "openvino/core/type/element_type_info.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/util/common_util.hpp"

namespace ov::element {
namespace {
constexpr size_t idx(Type_t e) noexcept {
    return static_cast<std::underlying_type_t<Type_t>>(e);
}

// Update it when new type is added
constexpr size_t enum_types_size = idx(gguf_tq2_0) + 1;

template <class Array>
constexpr TypeInfo type_info(size_t bitwidth,
                             bool is_real,
                             bool is_signed,
                             bool is_quantized,
                             const char* cname,
                             const char* type_name,
                             const Array& aliases) {
    return {bitwidth, is_real, is_signed, is_quantized, cname, type_name, aliases.data(), aliases.size()};
}

constexpr auto dynamic_aliases = util::make_array("UNSPECIFIED", "undefined");
constexpr auto boolean_aliases = util::make_array("BOOL");
constexpr auto bf16_aliases = util::make_array("BF16");
constexpr auto f16_aliases = util::make_array("FP16");
constexpr auto f32_aliases = util::make_array("FP32");
constexpr auto f64_aliases = util::make_array("FP64");
constexpr auto i4_aliases = util::make_array("I4");
constexpr auto i8_aliases = util::make_array("I8");
constexpr auto i16_aliases = util::make_array("I16");
constexpr auto i32_aliases = util::make_array("I32");
constexpr auto i64_aliases = util::make_array("I64");
constexpr auto u1_aliases = util::make_array("U1", "bin", "BIN");
constexpr auto u2_aliases = util::make_array("U2");
constexpr auto u3_aliases = util::make_array("U3");
constexpr auto u4_aliases = util::make_array("U4");
constexpr auto u6_aliases = util::make_array("U6");
constexpr auto u8_aliases = util::make_array("U8");
constexpr auto u16_aliases = util::make_array("U16");
constexpr auto u32_aliases = util::make_array("U32");
constexpr auto u64_aliases = util::make_array("U64");
constexpr auto nf4_aliases = util::make_array("NF4");
constexpr auto f8e4m3_aliases = util::make_array("F8E4M3");
constexpr auto f8e5m2_aliases = util::make_array("F8E5M2");
constexpr auto string_aliases = util::make_array("STRING");
constexpr auto f4e2m1_aliases = util::make_array("F4E2M1");
constexpr auto f8e8m0_aliases = util::make_array("F8E8M0");
constexpr auto gguf_q4_0_aliases = util::make_array("GGUF_Q4_0");
constexpr auto gguf_q4_1_aliases = util::make_array("GGUF_Q4_1");
constexpr auto gguf_q5_0_aliases = util::make_array("GGUF_Q5_0");
constexpr auto gguf_q5_1_aliases = util::make_array("GGUF_Q5_1");
constexpr auto gguf_q8_0_aliases = util::make_array("GGUF_Q8_0");
constexpr auto gguf_q8_1_aliases = util::make_array("GGUF_Q8_1");
constexpr auto gguf_q2_k_aliases = util::make_array("GGUF_Q2_K");
constexpr auto gguf_q3_k_aliases = util::make_array("GGUF_Q3_K");
constexpr auto gguf_q4_k_aliases = util::make_array("GGUF_Q4_K");
constexpr auto gguf_q5_k_aliases = util::make_array("GGUF_Q5_K");
constexpr auto gguf_q6_k_aliases = util::make_array("GGUF_Q6_K");
constexpr auto gguf_q8_k_aliases = util::make_array("GGUF_Q8_K");
constexpr auto gguf_iq2_xxs_aliases = util::make_array("GGUF_IQ2_XXS");
constexpr auto gguf_iq2_xs_aliases = util::make_array("GGUF_IQ2_XS");
constexpr auto gguf_iq3_xxs_aliases = util::make_array("GGUF_IQ3_XXS");
constexpr auto gguf_iq1_s_aliases = util::make_array("GGUF_IQ1_S");
constexpr auto gguf_iq4_nl_aliases = util::make_array("GGUF_IQ4_NL");
constexpr auto gguf_iq3_s_aliases = util::make_array("GGUF_IQ3_S");
constexpr auto gguf_iq2_s_aliases = util::make_array("GGUF_IQ2_S");
constexpr auto gguf_iq4_xs_aliases = util::make_array("GGUF_IQ4_XS");
constexpr auto gguf_iq1_m_aliases = util::make_array("GGUF_IQ1_M");
constexpr auto gguf_tq1_0_aliases = util::make_array("GGUF_TQ1_0");
constexpr auto gguf_tq2_0_aliases = util::make_array("GGUF_TQ2_0");

static constexpr std::array<TypeInfo, enum_types_size> types_info = {
    type_info(0, false, false, false, "dynamic", "dynamic", dynamic_aliases),                     // dynamic
    type_info(8, false, true, false, "char", "boolean", boolean_aliases),                         // boolean
    type_info(16, true, true, false, "bfloat16", "bf16", bf16_aliases),                           // bf16
    type_info(16, true, true, false, "float16", "f16", f16_aliases),                              // f16
    type_info(32, true, true, false, "float", "f32", f32_aliases),                                // f32
    type_info(64, true, true, false, "double", "f64", f64_aliases),                               // f64
    type_info(4, false, true, true, "int4_t", "i4", i4_aliases),                                  // i4
    type_info(8, false, true, true, "int8_t", "i8", i8_aliases),                                  // i8
    type_info(16, false, true, false, "int16_t", "i16", i16_aliases),                             // i16
    type_info(32, false, true, true, "int32_t", "i32", i32_aliases),                              // i32
    type_info(64, false, true, false, "int64_t", "i64", i64_aliases),                             // i64
    type_info(1, false, false, false, "uint1_t", "u1", u1_aliases),                               // u1
    type_info(2, false, false, false, "uint2_t", "u2", u2_aliases),                               // u2
    type_info(3, false, false, false, "uint3_t", "u3", u3_aliases),                               // u3
    type_info(4, false, false, false, "uint4_t", "u4", u4_aliases),                               // u4
    type_info(6, false, false, false, "uint6_t", "u6", u6_aliases),                               // u6
    type_info(8, false, false, true, "uint8_t", "u8", u8_aliases),                                // u8
    type_info(16, false, false, false, "uint16_t", "u16", u16_aliases),                           // u16
    type_info(32, false, false, false, "uint32_t", "u32", u32_aliases),                           // u32
    type_info(64, false, false, false, "uint64_t", "u64", u64_aliases),                           // u64
    type_info(4, false, false, true, "nfloat4", "nf4", nf4_aliases),                              // nf4
    type_info(8, true, true, true, "f8e4m3", "f8e4m3", f8e4m3_aliases),                           // f8e4m3
    type_info(8, true, true, true, "f8e5m2", "f8e5m2", f8e5m2_aliases),                           // f8e5m2
    type_info(8 * sizeof(std::string), false, false, false, "string", "string", string_aliases),  // string
    type_info(4, true, true, true, "f4e2m1", "f4e2m1", f4e2m1_aliases),                           // f4e2m1
    type_info(8, true, true, true, "f8e8m0", "f8e8m0", f8e8m0_aliases),                           // f8e8m0
    // GGUF block types: opaque blocks of bytes. m_bitwidth is the rounded-up average bits per
    // logical element (block_byte_size * 8 / block_elem_count) kept only for diagnostics; real
    // storage size is computed from block geometry in ov::util::get_memory_size. is_real=false,
    // is_quantized=true; is_signed follows the GGML block layout (see SPEC.md §1.2).
    type_info(5, false, true, true, "gguf_q4_0", "gguf_q4_0", gguf_q4_0_aliases),                  // gguf_q4_0
    type_info(5, false, false, true, "gguf_q4_1", "gguf_q4_1", gguf_q4_1_aliases),                 // gguf_q4_1
    type_info(6, false, true, true, "gguf_q5_0", "gguf_q5_0", gguf_q5_0_aliases),                  // gguf_q5_0
    type_info(6, false, false, true, "gguf_q5_1", "gguf_q5_1", gguf_q5_1_aliases),                 // gguf_q5_1
    type_info(9, false, true, true, "gguf_q8_0", "gguf_q8_0", gguf_q8_0_aliases),                  // gguf_q8_0
    type_info(9, false, false, true, "gguf_q8_1", "gguf_q8_1", gguf_q8_1_aliases),                 // gguf_q8_1
    type_info(3, false, false, true, "gguf_q2_k", "gguf_q2_k", gguf_q2_k_aliases),                 // gguf_q2_k
    type_info(4, false, true, true, "gguf_q3_k", "gguf_q3_k", gguf_q3_k_aliases),                  // gguf_q3_k
    type_info(5, false, false, true, "gguf_q4_k", "gguf_q4_k", gguf_q4_k_aliases),                 // gguf_q4_k
    type_info(6, false, false, true, "gguf_q5_k", "gguf_q5_k", gguf_q5_k_aliases),                 // gguf_q5_k
    type_info(7, false, true, true, "gguf_q6_k", "gguf_q6_k", gguf_q6_k_aliases),                  // gguf_q6_k
    type_info(10, false, true, true, "gguf_q8_k", "gguf_q8_k", gguf_q8_k_aliases),                 // gguf_q8_k
    type_info(3, false, true, true, "gguf_iq2_xxs", "gguf_iq2_xxs", gguf_iq2_xxs_aliases),         // gguf_iq2_xxs
    type_info(3, false, true, true, "gguf_iq2_xs", "gguf_iq2_xs", gguf_iq2_xs_aliases),            // gguf_iq2_xs
    type_info(4, false, true, true, "gguf_iq3_xxs", "gguf_iq3_xxs", gguf_iq3_xxs_aliases),         // gguf_iq3_xxs
    type_info(2, false, true, true, "gguf_iq1_s", "gguf_iq1_s", gguf_iq1_s_aliases),               // gguf_iq1_s
    type_info(5, false, true, true, "gguf_iq4_nl", "gguf_iq4_nl", gguf_iq4_nl_aliases),            // gguf_iq4_nl
    type_info(4, false, true, true, "gguf_iq3_s", "gguf_iq3_s", gguf_iq3_s_aliases),               // gguf_iq3_s
    type_info(3, false, true, true, "gguf_iq2_s", "gguf_iq2_s", gguf_iq2_s_aliases),               // gguf_iq2_s
    type_info(5, false, true, true, "gguf_iq4_xs", "gguf_iq4_xs", gguf_iq4_xs_aliases),            // gguf_iq4_xs
    type_info(2, false, true, true, "gguf_iq1_m", "gguf_iq1_m", gguf_iq1_m_aliases),               // gguf_iq1_m
    type_info(2, false, true, true, "gguf_tq1_0", "gguf_tq1_0", gguf_tq1_0_aliases),               // gguf_tq1_0
    type_info(3, false, true, true, "gguf_tq2_0", "gguf_tq2_0", gguf_tq2_0_aliases)                // gguf_tq2_0
};

constexpr bool validate_types_info(decltype(types_info)& info, size_t i = 0) {
    return i >= info.size() ? true : info[i].is_valid() ? validate_types_info(info, i + 1) : false;
}

static_assert(validate_types_info(types_info), "Some entries of type_info are invalid.");

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

Type type_from_string(const std::string& type) {
    const auto type_idx = type_idx_for(type);
    OPENVINO_ASSERT(is_valid_type_idx(type_idx), "Unsupported element type: ", type);
    return {static_cast<Type_t>(type_idx)};
}

// generate known types automatically
static constexpr auto known_types = [] {
    std::array<Type, enum_types_size - 1> types;
    for (size_t idx = 1, i = 0; i < types.size(); ++idx, ++i) {
        types[i] = Type{static_cast<Type_t>(idx)};
    }
    return types;
}();
}  // namespace

const TypeInfo& get_type_info(Type_t type) {
    const auto type_idx = idx(type);
    OPENVINO_ASSERT(is_valid_type_idx(type_idx), "Type_t not supported: ", type_idx);
    return types_info[type_idx];
}

std::vector<const Type*> Type::get_known_types() {
    std::vector<const Type*> result(known_types.size());
    for (size_t i = 0; i < known_types.size(); ++i) {
        result[i] = &known_types[i];
    }
    return result;
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

std::ostream& operator<<(std::ostream& out, const Type& obj) {
    return out << obj.to_string();
}

std::istream& operator>>(std::istream& in, Type& obj) {
    std::string str;
    in >> str;
    obj = Type(str);
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
    return m_type != Type_t::dynamic;
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

size_t Type::block_byte_size() const noexcept {
    return gguf_block_byte_size(m_type);
}

size_t Type::block_elem_count() const noexcept {
    return gguf_block_elem_count(m_type);
}

bool Type::is_gguf_block() const noexcept {
    return element::is_gguf_block(m_type);
}
}  // namespace ov::element

namespace ov {
template <>
OPENVINO_API EnumNames<element::Type_t>& EnumNames<element::Type_t>::get() {
    static auto enum_names =
        EnumNames<element::Type_t>("element::Type_t",
                                   {{"dynamic", element::Type_t::dynamic}, {"undefined", element::Type_t::dynamic},
                                    {"boolean", element::Type_t::boolean}, {"bf16", element::Type_t::bf16},
                                    {"f16", element::Type_t::f16},         {"f32", element::Type_t::f32},
                                    {"f64", element::Type_t::f64},         {"i4", element::Type_t::i4},
                                    {"i8", element::Type_t::i8},           {"i16", element::Type_t::i16},
                                    {"i32", element::Type_t::i32},         {"i64", element::Type_t::i64},
                                    {"u1", element::Type_t::u1},           {"u2", element::Type_t::u2},
                                    {"u3", element::Type_t::u3},           {"u4", element::Type_t::u4},
                                    {"u6", element::Type_t::u6},           {"u8", element::Type_t::u8},
                                    {"u16", element::Type_t::u16},         {"u32", element::Type_t::u32},
                                    {"u64", element::Type_t::u64},         {"nf4", element::Type_t::nf4},
                                    {"f8e4m3", element::Type_t::f8e4m3},   {"f8e5m2", element::Type_t::f8e5m2},
                                    {"string", element::Type_t::string},   {"f4e2m1", element::Type_t::f4e2m1},
                                    {"f8e8m0", element::Type_t::f8e8m0},
                                    {"gguf_q4_0", element::Type_t::gguf_q4_0},
                                    {"gguf_q4_1", element::Type_t::gguf_q4_1},
                                    {"gguf_q5_0", element::Type_t::gguf_q5_0},
                                    {"gguf_q5_1", element::Type_t::gguf_q5_1},
                                    {"gguf_q8_0", element::Type_t::gguf_q8_0},
                                    {"gguf_q8_1", element::Type_t::gguf_q8_1},
                                    {"gguf_q2_k", element::Type_t::gguf_q2_k},
                                    {"gguf_q3_k", element::Type_t::gguf_q3_k},
                                    {"gguf_q4_k", element::Type_t::gguf_q4_k},
                                    {"gguf_q5_k", element::Type_t::gguf_q5_k},
                                    {"gguf_q6_k", element::Type_t::gguf_q6_k},
                                    {"gguf_q8_k", element::Type_t::gguf_q8_k},
                                    {"gguf_iq2_xxs", element::Type_t::gguf_iq2_xxs},
                                    {"gguf_iq2_xs", element::Type_t::gguf_iq2_xs},
                                    {"gguf_iq3_xxs", element::Type_t::gguf_iq3_xxs},
                                    {"gguf_iq1_s", element::Type_t::gguf_iq1_s},
                                    {"gguf_iq4_nl", element::Type_t::gguf_iq4_nl},
                                    {"gguf_iq3_s", element::Type_t::gguf_iq3_s},
                                    {"gguf_iq2_s", element::Type_t::gguf_iq2_s},
                                    {"gguf_iq4_xs", element::Type_t::gguf_iq4_xs},
                                    {"gguf_iq1_m", element::Type_t::gguf_iq1_m},
                                    {"gguf_tq1_0", element::Type_t::gguf_tq1_0},
                                    {"gguf_tq2_0", element::Type_t::gguf_tq2_0}});
    return enum_names;
}

const std::string& AttributeAdapter<element::Type>::get() {
    return as_string(static_cast<element::Type_t>(m_ref));
}

void AttributeAdapter<element::Type>::set(const std::string& value) {
    m_ref = as_enum<element::Type_t>(value);
}

AttributeAdapter<ov::element::Type_t>::~AttributeAdapter() = default;
AttributeAdapter<ov::element::Type>::~AttributeAdapter() = default;
AttributeAdapter<ov::element::TypeVector>::~AttributeAdapter() = default;
}  // namespace ov
