// Copyright (C) 2018-2026 Intel Corporation
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
#include "openvino/core/except.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/core/type/float4_e2m1.hpp"
#include "openvino/core/type/float8_e4m3.hpp"
#include "openvino/core/type/float8_e5m2.hpp"
#include "openvino/core/type/float8_e8m0.hpp"

/**
 * @defgroup ov_element_cpp_api Element types
 * @ingroup ov_model_cpp_api
 * OpenVINO Element API to work with OpenVINO element types
 *
 */

namespace ov {
namespace element {
/// \brief Enum to define possible element types
/// \ingroup ov_element_cpp_api
enum class Type_t {
    dynamic,  //!< Dynamic element type
    boolean,  //!< boolean element type
    bf16,     //!< bf16 element type
    f16,      //!< f16 element type
    f32,      //!< f32 element type
    f64,      //!< f64 element type
    i4,       //!< i4 element type
    i8,       //!< i8 element type
    i16,      //!< i16 element type
    i32,      //!< i32 element type
    i64,      //!< i64 element type
    u1,       //!< binary element type
    u2,       //!< u2 element type
    u3,       //!< u3 element type
    u4,       //!< u4 element type
    u6,       //!< u6 element type
    u8,       //!< u8 element type
    u16,      //!< u16 element type
    u32,      //!< u32 element type
    u64,      //!< u64 element type
    nf4,      //!< nf4 element type
    f8e4m3,   //!< f8e4m3 element type
    f8e5m2,   //!< f8e5m2 element type
    string,   //!< string element type
    f4e2m1,   //!< f4e2m1 element type
    f8e8m0,   //!< f8e8m0 element type

    // GGUF block (llama.cpp) weight-only quantization types. Each value is an
    // opaque, self-describing block-of-bytes type whose only consumer is the GPU
    // FullyConnected GGUF implementation. Order matches the GGML ggml_type enum
    // (https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h).
    // Do NOT reorder or remove entries: these values are ABI surface.
    gguf_q4_0,     //!< GGUF Q4_0 block type
    gguf_q4_1,     //!< GGUF Q4_1 block type
    gguf_q5_0,     //!< GGUF Q5_0 block type
    gguf_q5_1,     //!< GGUF Q5_1 block type
    gguf_q8_0,     //!< GGUF Q8_0 block type
    gguf_q8_1,     //!< GGUF Q8_1 block type
    gguf_q2_k,     //!< GGUF Q2_K block type
    gguf_q3_k,     //!< GGUF Q3_K block type
    gguf_q4_k,     //!< GGUF Q4_K block type
    gguf_q5_k,     //!< GGUF Q5_K block type
    gguf_q6_k,     //!< GGUF Q6_K block type
    gguf_q8_k,     //!< GGUF Q8_K block type
    gguf_iq2_xxs,  //!< GGUF IQ2_XXS block type
    gguf_iq2_xs,   //!< GGUF IQ2_XS block type
    gguf_iq3_xxs,  //!< GGUF IQ3_XXS block type
    gguf_iq1_s,    //!< GGUF IQ1_S block type
    gguf_iq4_nl,   //!< GGUF IQ4_NL block type
    gguf_iq3_s,    //!< GGUF IQ3_S block type
    gguf_iq2_s,    //!< GGUF IQ2_S block type
    gguf_iq4_xs,   //!< GGUF IQ4_XS block type
    gguf_iq1_m,    //!< GGUF IQ1_M block type
    gguf_tq1_0,    //!< GGUF TQ1_0 block type
    gguf_tq2_0,    //!< GGUF TQ2_0 block type
};

/// \brief Base class to define element type
/// \ingroup ov_element_cpp_api
class OPENVINO_API Type {
public:
    constexpr Type() = default;
    constexpr Type(const Type&) = default;
    constexpr Type(const Type_t t) : m_type{t} {}
    explicit Type(const std::string& type);
    constexpr Type& operator=(const Type&) = default;
    std::string c_type_string() const;
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

    /// \brief Number of bytes occupied by a single GGUF block of this type.
    /// \return Block byte size for GGUF block types, 0 for any other type.
    size_t block_byte_size() const noexcept;
    /// \brief Number of logical weight elements packed into a single GGUF block.
    /// \return Block element count for GGUF block types, 0 for any other type.
    size_t block_elem_count() const noexcept;
    /// \brief Whether this element type is one of the opaque GGUF block types.
    bool is_gguf_block() const noexcept;
    // The name of this type, the enum name of this type
    std::string get_type_name() const;
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
    // Return element type in string representation
    std::string to_string() const;

private:
    Type_t m_type{Type_t::dynamic};
};

using TypeVector = std::vector<Type>;

/// \brief dynamic element type
/// \ingroup ov_element_cpp_api
inline constexpr Type dynamic(Type_t::dynamic);
/// \brief boolean element type
/// \ingroup ov_element_cpp_api
inline constexpr Type boolean(Type_t::boolean);
/// \brief bf16 element type
/// \ingroup ov_element_cpp_api
inline constexpr Type bf16(Type_t::bf16);
/// \brief f16 element type
/// \ingroup ov_element_cpp_api
inline constexpr Type f16(Type_t::f16);
/// \brief f32 element type
/// \ingroup ov_element_cpp_api
inline constexpr Type f32(Type_t::f32);
/// \brief f64 element type
/// \ingroup ov_element_cpp_api
inline constexpr Type f64(Type_t::f64);
/// \brief i4 element type
/// \ingroup ov_element_cpp_api
inline constexpr Type i4(Type_t::i4);
/// \brief i8 element type
/// \ingroup ov_element_cpp_api
inline constexpr Type i8(Type_t::i8);
/// \brief i16 element type
/// \ingroup ov_element_cpp_api
inline constexpr Type i16(Type_t::i16);
/// \brief i32 element type
/// \ingroup ov_element_cpp_api
inline constexpr Type i32(Type_t::i32);
/// \brief i64 element type
/// \ingroup ov_element_cpp_api
inline constexpr Type i64(Type_t::i64);
/// \brief binary element type
/// \ingroup ov_element_cpp_api
inline constexpr Type u1(Type_t::u1);
/// \brief u2 element type
/// \ingroup ov_element_cpp_api
inline constexpr Type u2(Type_t::u2);
/// \brief u3 element type
/// \ingroup ov_element_cpp_api
inline constexpr Type u3(Type_t::u3);
/// \brief u4 element type
/// \ingroup ov_element_cpp_api
inline constexpr Type u4(Type_t::u4);
/// \brief u6 element type
/// \ingroup ov_element_cpp_api
inline constexpr Type u6(Type_t::u6);
/// \brief u8 element type
/// \ingroup ov_element_cpp_api
inline constexpr Type u8(Type_t::u8);
/// \brief u16 element type
/// \ingroup ov_element_cpp_api
inline constexpr Type u16(Type_t::u16);
/// \brief u32 element type
/// \ingroup ov_element_cpp_api
inline constexpr Type u32(Type_t::u32);
/// \brief u64 element type
/// \ingroup ov_element_cpp_api
inline constexpr Type u64(Type_t::u64);
/// \brief nf4 element type
/// \ingroup ov_element_cpp_api
inline constexpr Type nf4(Type_t::nf4);
/// \brief f8e4m3 element type
/// \ingroup ov_element_cpp_api
inline constexpr Type f8e4m3(Type_t::f8e4m3);
/// \brief f8e4m3 element type
/// \ingroup ov_element_cpp_api
inline constexpr Type f8e5m2(Type_t::f8e5m2);
/// \brief string element type
/// \ingroup ov_element_cpp_api
inline constexpr Type string(Type_t::string);
/// \brief f4e2m1 element type
/// \ingroup ov_element_cpp_api
inline constexpr Type f4e2m1(Type_t::f4e2m1);
/// \brief f8e8m0 element type
/// \ingroup ov_element_cpp_api
inline constexpr Type f8e8m0(Type_t::f8e8m0);
/// \brief GGUF Q4_0 block element type
/// \ingroup ov_element_cpp_api
inline constexpr Type gguf_q4_0(Type_t::gguf_q4_0);
/// \brief GGUF Q4_1 block element type
/// \ingroup ov_element_cpp_api
inline constexpr Type gguf_q4_1(Type_t::gguf_q4_1);
/// \brief GGUF Q5_0 block element type
/// \ingroup ov_element_cpp_api
inline constexpr Type gguf_q5_0(Type_t::gguf_q5_0);
/// \brief GGUF Q5_1 block element type
/// \ingroup ov_element_cpp_api
inline constexpr Type gguf_q5_1(Type_t::gguf_q5_1);
/// \brief GGUF Q8_0 block element type
/// \ingroup ov_element_cpp_api
inline constexpr Type gguf_q8_0(Type_t::gguf_q8_0);
/// \brief GGUF Q8_1 block element type
/// \ingroup ov_element_cpp_api
inline constexpr Type gguf_q8_1(Type_t::gguf_q8_1);
/// \brief GGUF Q2_K block element type
/// \ingroup ov_element_cpp_api
inline constexpr Type gguf_q2_k(Type_t::gguf_q2_k);
/// \brief GGUF Q3_K block element type
/// \ingroup ov_element_cpp_api
inline constexpr Type gguf_q3_k(Type_t::gguf_q3_k);
/// \brief GGUF Q4_K block element type
/// \ingroup ov_element_cpp_api
inline constexpr Type gguf_q4_k(Type_t::gguf_q4_k);
/// \brief GGUF Q5_K block element type
/// \ingroup ov_element_cpp_api
inline constexpr Type gguf_q5_k(Type_t::gguf_q5_k);
/// \brief GGUF Q6_K block element type
/// \ingroup ov_element_cpp_api
inline constexpr Type gguf_q6_k(Type_t::gguf_q6_k);
/// \brief GGUF Q8_K block element type
/// \ingroup ov_element_cpp_api
inline constexpr Type gguf_q8_k(Type_t::gguf_q8_k);
/// \brief GGUF IQ2_XXS block element type
/// \ingroup ov_element_cpp_api
inline constexpr Type gguf_iq2_xxs(Type_t::gguf_iq2_xxs);
/// \brief GGUF IQ2_XS block element type
/// \ingroup ov_element_cpp_api
inline constexpr Type gguf_iq2_xs(Type_t::gguf_iq2_xs);
/// \brief GGUF IQ3_XXS block element type
/// \ingroup ov_element_cpp_api
inline constexpr Type gguf_iq3_xxs(Type_t::gguf_iq3_xxs);
/// \brief GGUF IQ1_S block element type
/// \ingroup ov_element_cpp_api
inline constexpr Type gguf_iq1_s(Type_t::gguf_iq1_s);
/// \brief GGUF IQ4_NL block element type
/// \ingroup ov_element_cpp_api
inline constexpr Type gguf_iq4_nl(Type_t::gguf_iq4_nl);
/// \brief GGUF IQ3_S block element type
/// \ingroup ov_element_cpp_api
inline constexpr Type gguf_iq3_s(Type_t::gguf_iq3_s);
/// \brief GGUF IQ2_S block element type
/// \ingroup ov_element_cpp_api
inline constexpr Type gguf_iq2_s(Type_t::gguf_iq2_s);
/// \brief GGUF IQ4_XS block element type
/// \ingroup ov_element_cpp_api
inline constexpr Type gguf_iq4_xs(Type_t::gguf_iq4_xs);
/// \brief GGUF IQ1_M block element type
/// \ingroup ov_element_cpp_api
inline constexpr Type gguf_iq1_m(Type_t::gguf_iq1_m);
/// \brief GGUF TQ1_0 block element type
/// \ingroup ov_element_cpp_api
inline constexpr Type gguf_tq1_0(Type_t::gguf_tq1_0);
/// \brief GGUF TQ2_0 block element type
/// \ingroup ov_element_cpp_api
inline constexpr Type gguf_tq2_0(Type_t::gguf_tq2_0);

/// \brief Number of logical weight elements in a single GGUF block of the given type.
/// \return Block element count for GGUF block types (32 or 256), 0 otherwise.
/// \ingroup ov_element_cpp_api
constexpr size_t gguf_block_elem_count(Type_t t) noexcept {
    switch (t) {
    case Type_t::gguf_q4_0:
    case Type_t::gguf_q4_1:
    case Type_t::gguf_q5_0:
    case Type_t::gguf_q5_1:
    case Type_t::gguf_q8_0:
    case Type_t::gguf_q8_1:
    case Type_t::gguf_iq4_nl:
        return 32;
    case Type_t::gguf_q2_k:
    case Type_t::gguf_q3_k:
    case Type_t::gguf_q4_k:
    case Type_t::gguf_q5_k:
    case Type_t::gguf_q6_k:
    case Type_t::gguf_q8_k:
    case Type_t::gguf_iq2_xxs:
    case Type_t::gguf_iq2_xs:
    case Type_t::gguf_iq3_xxs:
    case Type_t::gguf_iq1_s:
    case Type_t::gguf_iq3_s:
    case Type_t::gguf_iq2_s:
    case Type_t::gguf_iq4_xs:
    case Type_t::gguf_iq1_m:
    case Type_t::gguf_tq1_0:
    case Type_t::gguf_tq2_0:
        return 256;
    default:
        return 0;
    }
}

/// \brief Number of bytes occupied by a single GGUF block of the given type.
/// \return Block byte size for GGUF block types, 0 otherwise.
/// \ingroup ov_element_cpp_api
constexpr size_t gguf_block_byte_size(Type_t t) noexcept {
    switch (t) {
    case Type_t::gguf_q4_0:
        return 18;
    case Type_t::gguf_q4_1:
        return 20;
    case Type_t::gguf_q5_0:
        return 22;
    case Type_t::gguf_q5_1:
        return 24;
    case Type_t::gguf_q8_0:
        return 34;
    case Type_t::gguf_q8_1:
        return 36;
    case Type_t::gguf_q2_k:
        return 84;
    case Type_t::gguf_q3_k:
        return 110;
    case Type_t::gguf_q4_k:
        return 144;
    case Type_t::gguf_q5_k:
        return 176;
    case Type_t::gguf_q6_k:
        return 210;
    case Type_t::gguf_q8_k:
        return 292;
    case Type_t::gguf_iq2_xxs:
        return 66;
    case Type_t::gguf_iq2_xs:
        return 74;
    case Type_t::gguf_iq3_xxs:
        return 98;
    case Type_t::gguf_iq1_s:
        return 50;
    case Type_t::gguf_iq4_nl:
        return 18;
    case Type_t::gguf_iq3_s:
        return 110;
    case Type_t::gguf_iq2_s:
        return 82;
    case Type_t::gguf_iq4_xs:
        return 136;
    case Type_t::gguf_iq1_m:
        return 56;
    case Type_t::gguf_tq1_0:
        return 54;
    case Type_t::gguf_tq2_0:
        return 66;
    default:
        return 0;
    }
}

/// \brief Whether the given element type is one of the opaque GGUF block types.
/// \ingroup ov_element_cpp_api
constexpr bool is_gguf_block(Type_t t) noexcept {
    return gguf_block_elem_count(t) != 0;
}

/// \brief Whether the given element type is one of the opaque GGUF block types.
/// \ingroup ov_element_cpp_api
inline bool is_gguf_block(const Type& t) noexcept {
    return is_gguf_block(static_cast<Type_t>(t));
}

template <class T>
constexpr Type from() {
    if constexpr (std::is_same_v<T, char> || std::is_same_v<T, bool>) {
        return boolean;
    } else if constexpr (std::is_same_v<T, ov::float16>) {
        return f16;
    } else if constexpr (std::is_same_v<T, float>) {
        return f32;
    } else if constexpr (std::is_same_v<T, double>) {
        return f64;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return i8;
    } else if constexpr (std::is_same_v<T, int16_t>) {
        return i16;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return i32;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return i64;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return u8;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return u16;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return u32;
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return u64;
    } else if constexpr (std::is_same_v<T, ov::bfloat16>) {
        return bf16;
    } else if constexpr (std::is_same_v<T, ov::float8_e4m3>) {
        return f8e4m3;
    } else if constexpr (std::is_same_v<T, ov::float8_e5m2>) {
        return f8e5m2;
    } else if constexpr (std::is_same_v<T, std::string>) {
        return string;
    } else if constexpr (std::is_same_v<T, ov::float4_e2m1>) {
        return f4e2m1;
    } else if constexpr (std::is_same_v<T, ov::float8_e8m0>) {
        return f8e8m0;
    } else {
        OPENVINO_THROW("Unknown type");
    }
}

OPENVINO_API
std::ostream& operator<<(std::ostream& out, const ov::element::Type& obj);

OPENVINO_API
std::istream& operator>>(std::istream& out, ov::element::Type& obj);
}  // namespace element

template <>
class OPENVINO_API AttributeAdapter<ov::element::Type_t> : public EnumAttributeAdapterBase<ov::element::Type_t> {
public:
    AttributeAdapter(ov::element::Type_t& value) : EnumAttributeAdapterBase<ov::element::Type_t>(value) {}
    ~AttributeAdapter() override;

    OPENVINO_RTTI("AttributeAdapter<ov::element::Type_t>");
};

template <>
class OPENVINO_API AttributeAdapter<ov::element::Type> : public ValueAccessor<std::string> {
public:
    OPENVINO_RTTI("AttributeAdapter<ov::element::Type>");
    constexpr AttributeAdapter(ov::element::Type& value) : m_ref(value) {}
    ~AttributeAdapter() override;

    const std::string& get() override;
    void set(const std::string& value) override;

    constexpr operator ov::element::Type&() {
        return m_ref;
    }

protected:
    ov::element::Type& m_ref;
};

template <>
class OPENVINO_API AttributeAdapter<ov::element::TypeVector> : public DirectValueAccessor<ov::element::TypeVector> {
public:
    OPENVINO_RTTI("AttributeAdapter<ov::element::TypeVector>");
    AttributeAdapter(ov::element::TypeVector& value) : DirectValueAccessor<ov::element::TypeVector>(value) {}
    ~AttributeAdapter() override;
};

}  // namespace ov
