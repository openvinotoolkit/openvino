//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

//================================================================================================
// ElementType
//================================================================================================

#pragma once

#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/deprecated.hpp"
#include "ngraph/except.hpp"
#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/type/bfloat16.hpp"
#include "ngraph/type/float16.hpp"

namespace ngraph
{
    namespace element
    {
        enum class Type_t
        {
            undefined,
            dynamic,
            boolean,
            bf16,
            f16,
            f32,
            f64,
            i8,
            i16,
            i32,
            i64,
            u1,
            u8,
            u16,
            u32,
            u64
        };

        class NGRAPH_API Type
        {
        public:
            Type()
                : m_type{element::Type_t::undefined}
            {
            }
            Type(const Type&) = default;
            constexpr Type(const Type_t t)
                : m_type{t}
            {
            }
            Type(size_t bitwidth,
                 bool is_real,
                 bool is_signed,
                 bool is_quantized,
                 const std::string& cname);
            Type& operator=(const Type&) = default;
            const std::string& c_type_string() const;
            size_t size() const;
            size_t hash() const;
            bool is_static() const;
            bool is_dynamic() const { return !is_static(); }
            bool is_real() const;
            // TODO: We may want to revisit this definition when we do a more general cleanup of
            // element types:
            bool is_integral() const { return !is_real(); }
            bool is_integral_number() const;
            bool is_signed() const;
            bool is_quantized() const;
            size_t bitwidth() const;
            // The name of this type, the enum name of this type
            const std::string& get_type_name() const;
            friend NGRAPH_API std::ostream& operator<<(std::ostream&, const Type&);
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
            constexpr operator Type_t() const { return m_type; }

        private:
            Type_t m_type{Type_t::undefined};
        };

        typedef std::vector<Type> TypeVector;

        constexpr Type undefined(Type_t::undefined);
        constexpr Type dynamic(Type_t::dynamic);
        constexpr Type boolean(Type_t::boolean);
        constexpr Type bf16(Type_t::bf16);
        constexpr Type f16(Type_t::f16);
        constexpr Type f32(Type_t::f32);
        constexpr Type f64(Type_t::f64);
        constexpr Type i8(Type_t::i8);
        constexpr Type i16(Type_t::i16);
        constexpr Type i32(Type_t::i32);
        constexpr Type i64(Type_t::i64);
        constexpr Type u1(Type_t::u1);
        constexpr Type u8(Type_t::u8);
        constexpr Type u16(Type_t::u16);
        constexpr Type u32(Type_t::u32);
        constexpr Type u64(Type_t::u64);

        template <typename T>
        Type from()
        {
            throw std::invalid_argument("Unknown type");
        }
        template <>
        NGRAPH_API Type from<char>();
        template <>
        NGRAPH_API Type from<bool>();
        template <>
        NGRAPH_API Type from<float>();
        template <>
        NGRAPH_API Type from<double>();
        template <>
        NGRAPH_API Type from<int8_t>();
        template <>
        NGRAPH_API Type from<int16_t>();
        template <>
        NGRAPH_API Type from<int32_t>();
        template <>
        NGRAPH_API Type from<int64_t>();
        template <>
        NGRAPH_API Type from<uint8_t>();
        template <>
        NGRAPH_API Type from<uint16_t>();
        template <>
        NGRAPH_API Type from<uint32_t>();
        template <>
        NGRAPH_API Type from<uint64_t>();
        template <>
        NGRAPH_API Type from<ngraph::bfloat16>();
        template <>
        NGRAPH_API Type from<ngraph::float16>();

        NGRAPH_API
        std::ostream& operator<<(std::ostream& out, const ngraph::element::Type& obj);
    }

    template <>
    class NGRAPH_API AttributeAdapter<element::Type_t>
        : public EnumAttributeAdapterBase<element::Type_t>
    {
    public:
        AttributeAdapter(element::Type_t& value)
            : EnumAttributeAdapterBase<element::Type_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<element::Type_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    template <>
    class NGRAPH_API AttributeAdapter<element::Type> : public ValueAccessor<std::string>
    {
    public:
        AttributeAdapter(element::Type& value)
            : m_ref(value)
        {
        }

        const std::string& get() override;
        void set(const std::string& value) override;

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<element::Type>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        operator element::Type&() { return m_ref; }

    protected:
        element::Type& m_ref;
    };

    /// \brief Return the number of bytes in the compile-time representation of the element type.
    size_t compiler_byte_size(element::Type_t et);
}
