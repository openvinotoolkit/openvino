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

#pragma once

#include <string>
#include <type_traits>
#include <vector>

#include "ngraph/enum_names.hpp"
#include "ngraph/type.hpp"

///
namespace ngraph
{
    class AttributeVisitor;

    /// \brief Provides access to an attribute of type AT as a value accessor type VAT
    template <typename VAT>
    class ValueAccessor;

    /// \brief ValueAccessor<void> provides an accessor for values that do not have get/set methonds
    /// via AttributeVistor.on_adapter.
    ///
    /// All ValueAccessors must be derived from ValueAccessor<void> so that an AttributeVisitor
    /// only needs to implement a subset of the on_adapter methods.
    template <>
    class NGRAPH_API ValueAccessor<void>
    {
    public:
        /// \brief type info enables identification of the value accessor, as well as is_type and
        /// as_type.
        virtual const DiscreteTypeInfo& get_type_info() const = 0;
        virtual ~ValueAccessor() {}
    };

    /// \brief Provides access to values via get/set methods from an m_value, typically from
    /// ValueReference
    ///
    /// The m_buffer holds a VAT, which may be wider than the attribute AT. For example, serializers
    /// that only
    /// support int64_t integers would use a ValueAccessor<vector<int64_t>> to reference a
    /// vector<int8_t> attribute. Destruction moves the value back to the attribute if it was
    /// changed.
    /// \tparam VAT The adapter value type; may be wider than the value being accessed.
    template <typename VAT>
    class ValueAccessor : public ValueAccessor<void>
    {
    public:
        /// Returns the value
        virtual const VAT& get() = 0;
        /// Sets the value
        virtual void set(const VAT& value) = 0;
    };

    template <>
    class ValueAccessor<void*> : public ValueAccessor<void>
    {
    public:
        virtual void* get_ptr() = 0;
        virtual size_t size() = 0;
    };

    template <typename AT>
    class DirectValueAccessor : public ValueAccessor<AT>
    {
    public:
        DirectValueAccessor(AT& ref)
            : m_ref(ref)
        {
        }
        const AT& get() override { return m_ref; }
        void set(const AT& value) override { m_ref = value; }

    protected:
        AT& m_ref;
    };

    template <typename AT, typename VAT>
    class IndirectScalarValueAccessor : public ValueAccessor<VAT>
    {
    public:
        IndirectScalarValueAccessor(AT& ref)
            : m_ref(ref)
            , m_buffer()
        {
        }

        const VAT& get() override
        {
            if (!m_buffer_valid)
            {
                m_buffer = static_cast<VAT>(m_ref);
                m_buffer_valid = true;
            }
            return m_buffer;
        }

        void set(const VAT& value) override
        {
            m_ref = static_cast<AT>(value);
            m_buffer_valid = false;
        }

    protected:
        AT& m_ref;
        VAT m_buffer;
        bool m_buffer_valid{false};
    };

    template <typename A, typename B>
    A copy_from(B& b)
    {
        A result(b.size());
        for (size_t i = 0; i < b.size(); ++i)
        {
            result[i] =
                static_cast<typename std::remove_reference<decltype(result[i])>::type>(b[i]);
        }
        return result;
    }

    template <typename AT, typename VAT>
    class IndirectVectorValueAccessor : public ValueAccessor<VAT>
    {
    public:
        IndirectVectorValueAccessor(AT& ref)
            : m_ref(ref)
        {
        }

        const VAT& get() override
        {
            if (!m_buffer_valid)
            {
                m_buffer = copy_from<typename std::remove_cv<VAT>::type>(m_ref);
                m_buffer_valid = true;
            }
            return m_buffer;
        }

        void set(const VAT& value) override
        {
            m_ref = copy_from<AT>(value);
            m_buffer_valid = false;
        }

        operator AT&() { return m_ref; }

    protected:
        AT& m_ref;
        VAT m_buffer;
        bool m_buffer_valid{false};
    };

    /// \brief An AttributeAdapter "captures" an attribute as an AT& and makes it available as a
    /// ValueAccessor<VAT>.
    template <typename AT>
    class AttributeAdapter
    {
    };

    /// \brief Access an enum via a string
    /// \tparam AT The attribute type enum class
    template <typename AT>
    class EnumAttributeAdapterBase : public ValueAccessor<std::string>
    {
    public:
        EnumAttributeAdapterBase(AT& value)
            : m_ref(value)
        {
        }

        const std::string& get() override { return as_string(m_ref); }
        void set(const std::string& value) override { m_ref = as_enum<AT>(value); }
        operator AT&() { return m_ref; }

    protected:
        AT& m_ref;
    };

    /// Adapters will see visitor
    class VisitorAdapter : public ValueAccessor<void>
    {
    public:
        virtual bool visit_attributes(AttributeVisitor& visitor) = 0;
    };

    template <>
    class NGRAPH_API AttributeAdapter<float> : public IndirectScalarValueAccessor<float, double>
    {
    public:
        AttributeAdapter(float& value)
            : IndirectScalarValueAccessor<float, double>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<float>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    /// \brief Access a double as a double
    template <>
    class NGRAPH_API AttributeAdapter<double> : public DirectValueAccessor<double>
    {
    public:
        AttributeAdapter(double& value)
            : DirectValueAccessor<double>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<double>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    /// \brief Access a string as a string
    template <>
    class NGRAPH_API AttributeAdapter<std::string> : public DirectValueAccessor<std::string>
    {
    public:
        AttributeAdapter(std::string& value)
            : DirectValueAccessor<std::string>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<string>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    /// \brief Access a bool as a bool
    template <>
    class NGRAPH_API AttributeAdapter<bool> : public DirectValueAccessor<bool>
    {
    public:
        AttributeAdapter(bool& value)
            : DirectValueAccessor<bool>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<bool>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    /// \brief Access an int8_t and an int64_t
    template <>
    class NGRAPH_API AttributeAdapter<int8_t> : public IndirectScalarValueAccessor<int8_t, int64_t>
    {
    public:
        AttributeAdapter(int8_t& value)
            : IndirectScalarValueAccessor<int8_t, int64_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<int8_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    /// \brief Access an int16_t as an int64_t
    template <>
    class NGRAPH_API AttributeAdapter<int16_t>
        : public IndirectScalarValueAccessor<int16_t, int64_t>
    {
    public:
        AttributeAdapter(int16_t& value)
            : IndirectScalarValueAccessor<int16_t, int64_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<int16_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    /// \brief Access an int32_t as an int64_t
    template <>
    class NGRAPH_API AttributeAdapter<int32_t>
        : public IndirectScalarValueAccessor<int32_t, int64_t>
    {
    public:
        AttributeAdapter(int32_t& value)
            : IndirectScalarValueAccessor<int32_t, int64_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<int32_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    /// \brief Access an int64_t as an int64_t
    template <>
    class NGRAPH_API AttributeAdapter<int64_t> : public DirectValueAccessor<int64_t>
    {
    public:
        AttributeAdapter(int64_t& value)
            : DirectValueAccessor<int64_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<int64_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    /// \brief Access a uint8_t as an int64_t
    template <>
    class NGRAPH_API AttributeAdapter<uint8_t>
        : public IndirectScalarValueAccessor<uint8_t, int64_t>
    {
    public:
        AttributeAdapter(uint8_t& value)
            : IndirectScalarValueAccessor<uint8_t, int64_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<uint8_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    /// \brief Access a uint16_t as an int64_t
    template <>
    class NGRAPH_API AttributeAdapter<uint16_t>
        : public IndirectScalarValueAccessor<uint16_t, int64_t>
    {
    public:
        AttributeAdapter(uint16_t& value)
            : IndirectScalarValueAccessor<uint16_t, int64_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<uint16_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    /// \brief Access a uint32_t as an int64_t
    template <>
    class NGRAPH_API AttributeAdapter<uint32_t>
        : public IndirectScalarValueAccessor<uint32_t, int64_t>
    {
    public:
        AttributeAdapter(uint32_t& value)
            : IndirectScalarValueAccessor<uint32_t, int64_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<uint32_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    /// \brief Access a uint64_t as an int64_t
    template <>
    class NGRAPH_API AttributeAdapter<uint64_t>
        : public IndirectScalarValueAccessor<uint64_t, int64_t>
    {
    public:
        AttributeAdapter(uint64_t& value)
            : IndirectScalarValueAccessor<uint64_t, int64_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<uint64_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

#ifdef __APPLE__
    // size_t is one of the uint types on _WIN32
    template <>
    class NGRAPH_API AttributeAdapter<size_t> : public IndirectScalarValueAccessor<size_t, int64_t>
    {
    public:
        AttributeAdapter(size_t& value)
            : IndirectScalarValueAccessor<size_t, int64_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<size_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    template <>
    class NGRAPH_API AttributeAdapter<std::vector<size_t>>
        : public IndirectVectorValueAccessor<std::vector<size_t>, std::vector<int64_t>>
    {
    public:
        AttributeAdapter(std::vector<size_t>& value)
            : IndirectVectorValueAccessor<std::vector<size_t>, std::vector<int64_t>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<vector<size_t>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };
#endif

    /// Note: These class bodies cannot be defined with templates because of interactions
    /// between dllexport and templates on Windows.

    /// \brief Access a vector<int8_t>
    template <>
    class NGRAPH_API AttributeAdapter<std::vector<int8_t>>
        : public DirectValueAccessor<std::vector<int8_t>>
    {
    public:
        AttributeAdapter(std::vector<int8_t>& value)
            : DirectValueAccessor<std::vector<int8_t>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<vector<int8_t>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    /// \brief Access a vector<int16_t>
    template <>
    class NGRAPH_API AttributeAdapter<std::vector<int16_t>>
        : public DirectValueAccessor<std::vector<int16_t>>
    {
    public:
        AttributeAdapter(std::vector<int16_t>& value)
            : DirectValueAccessor<std::vector<int16_t>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<vector<int16_t>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    /// \brief Access a vector<int32_t>
    template <>
    class NGRAPH_API AttributeAdapter<std::vector<int32_t>>
        : public DirectValueAccessor<std::vector<int32_t>>
    {
    public:
        AttributeAdapter(std::vector<int32_t>& value)
            : DirectValueAccessor<std::vector<int32_t>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<vector<int32_t>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    /// \brief Access a vector<int64_t>
    template <>
    class NGRAPH_API AttributeAdapter<std::vector<int64_t>>
        : public DirectValueAccessor<std::vector<int64_t>>
    {
    public:
        AttributeAdapter(std::vector<int64_t>& value)
            : DirectValueAccessor<std::vector<int64_t>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<vector<int64_t>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    /// \brief Access a vector<uint8_t>
    template <>
    class NGRAPH_API AttributeAdapter<std::vector<uint8_t>>
        : public DirectValueAccessor<std::vector<uint8_t>>
    {
    public:
        AttributeAdapter(std::vector<uint8_t>& value)
            : DirectValueAccessor<std::vector<uint8_t>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<vector<uint8_t>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    /// \brief Access a vector<uint16_t>
    template <>
    class NGRAPH_API AttributeAdapter<std::vector<uint16_t>>
        : public DirectValueAccessor<std::vector<uint16_t>>
    {
    public:
        AttributeAdapter(std::vector<uint16_t>& value)
            : DirectValueAccessor<std::vector<uint16_t>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<vector<uint16_t>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    /// \brief Access a vector<uint32_t>
    template <>
    class NGRAPH_API AttributeAdapter<std::vector<uint32_t>>
        : public DirectValueAccessor<std::vector<uint32_t>>
    {
    public:
        AttributeAdapter(std::vector<uint32_t>& value)
            : DirectValueAccessor<std::vector<uint32_t>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<vector<uint32_t>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    /// \brief Access a vector<uint64_t>
    template <>
    class NGRAPH_API AttributeAdapter<std::vector<uint64_t>>
        : public DirectValueAccessor<std::vector<uint64_t>>
    {
    public:
        AttributeAdapter(std::vector<uint64_t>& value)
            : DirectValueAccessor<std::vector<uint64_t>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<vector<uint64_t>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    /// \brief Access a vector<float>
    template <>
    class NGRAPH_API AttributeAdapter<std::vector<float>>
        : public DirectValueAccessor<std::vector<float>>
    {
    public:
        AttributeAdapter(std::vector<float>& value)
            : DirectValueAccessor<std::vector<float>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<vector<float>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    /// \brief Access a vector<double>
    template <>
    class NGRAPH_API AttributeAdapter<std::vector<double>>
        : public DirectValueAccessor<std::vector<double>>
    {
    public:
        AttributeAdapter(std::vector<double>& value)
            : DirectValueAccessor<std::vector<double>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<vector<double>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    /// \brief Access a vector<string>
    template <>
    class NGRAPH_API AttributeAdapter<std::vector<std::string>>
        : public DirectValueAccessor<std::vector<std::string>>
    {
    public:
        AttributeAdapter(std::vector<std::string>& value)
            : DirectValueAccessor<std::vector<std::string>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<vector<string>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };
}
