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

#pragma once

#include <string>
#include <type_traits>
#include <vector>

#include "ngraph/enum_names.hpp"
#include "ngraph/type.hpp"

namespace ngraph
{
    template <typename Type>
    class ValueAccessor;

    /// \brief ValueAccessor<void> provides an accessor for values that do not have get/set methonds
    template <>
    class NGRAPH_API ValueAccessor<void>
    {
    public:
        virtual const DiscreteTypeInfo& get_type_info() const = 0;
        virtual ~ValueAccessor() {}
    };

    /// \brief Provides access to values via get/set methods
    /// \tparam T The type of the value; may be wider than the value being accessed.
    template <typename T>
    class ValueAccessor : public ValueAccessor<void>
    {
    public:
        virtual const DiscreteTypeInfo& get_type_info() const = 0;
        /// Returns the value
        virtual const T& get() = 0;
        /// Sets the value
        virtual void set(const T& value) = 0;

    protected:
        T m_buffer;
        bool m_buffer_valid{false};
    };

    /// \brief holds a reference to a value
    /// \tparam Type the type of the referenced value
    template <typename Type>
    class ValueReference
    {
    public:
        operator Type&() const { return m_value; }
    protected:
        ValueReference(Type& value)
            : m_value(value)
        {
        }
        Type& m_value;
    };

    template <typename Type>
    class AttributeAdapter
    {
    };

    /// \brief Access an enum via a string
    /// \tparam Type The enum class
    template <typename Type>
    class EnumAttributeAdapterBase : public ValueReference<Type>, public ValueAccessor<std::string>
    {
    public:
        EnumAttributeAdapterBase(Type& value)
            : ValueReference<Type>(value)
        {
        }

        const std::string& get() override { return as_string(ValueReference<Type>::m_value); }
        void set(const std::string& value) override
        {
            ValueReference<Type>::m_value = as_enum<Type>(value);
        }
    };

    /// \brief Access a float as a double
    template <>
    class NGRAPH_API AttributeAdapter<float> : public ValueReference<float>,
                                               public ValueAccessor<double>
    {
    public:
        AttributeAdapter(float& value)
            : ValueReference<float>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<float>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const double& get() override;
        void set(const double& value) override;
    };

    /// \brief Access a double as a double
    template <>
    class NGRAPH_API AttributeAdapter<double> : public ValueReference<double>,
                                                public ValueAccessor<double>
    {
    public:
        AttributeAdapter(double& value)
            : ValueReference<double>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<double>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const double& get() override;
        void set(const double& value) override;
    };

    /// \brief Access a bool as a bool
    template <>
    class NGRAPH_API AttributeAdapter<bool> : public ValueReference<bool>,
                                              public ValueAccessor<bool>
    {
    public:
        AttributeAdapter(bool& value)
            : ValueReference<bool>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<bool>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const bool& get() override;
        void set(const bool& value) override;
    };

    /// \brief Access an int8_t and an int16_t
    template <>
    class NGRAPH_API AttributeAdapter<int8_t> : public ValueReference<int8_t>,
                                                public ValueAccessor<int64_t>
    {
    public:
        AttributeAdapter(int8_t& value)
            : ValueReference<int8_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<int8_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const int64_t& get() override;
        void set(const int64_t& value) override;
    };

    /// \brief Access an int16_t as an int64_t
    template <>
    class NGRAPH_API AttributeAdapter<int16_t> : public ValueReference<int16_t>,
                                                 public ValueAccessor<int64_t>
    {
    public:
        AttributeAdapter(int16_t& value)
            : ValueReference<int16_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<int16_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const int64_t& get() override;
        void set(const int64_t& value) override;
    };

    /// \brief Access an int32_t as an int64_t
    template <>
    class NGRAPH_API AttributeAdapter<int32_t> : public ValueReference<int32_t>,
                                                 public ValueAccessor<int64_t>
    {
    public:
        AttributeAdapter(int32_t& value)
            : ValueReference<int32_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<int32_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const int64_t& get() override;
        void set(const int64_t& value) override;
    };

    /// \brief Access an int64_t as an int64_t
    template <>
    class NGRAPH_API AttributeAdapter<int64_t> : public ValueReference<int64_t>,
                                                 public ValueAccessor<int64_t>
    {
    public:
        AttributeAdapter(int64_t& value)
            : ValueReference<int64_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<int64_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const int64_t& get() override;
        void set(const int64_t& value) override;
    };

    /// \brief Access a uint8_t as an int64_t
    template <>
    class NGRAPH_API AttributeAdapter<uint8_t> : public ValueReference<uint8_t>,
                                                 public ValueAccessor<int64_t>
    {
    public:
        AttributeAdapter(uint8_t& value)
            : ValueReference<uint8_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<uint8_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const int64_t& get() override;
        void set(const int64_t& value) override;
    };

    /// \brief Access a uint16_t as an int64_t
    template <>
    class NGRAPH_API AttributeAdapter<uint16_t> : public ValueReference<uint16_t>,
                                                  public ValueAccessor<int64_t>
    {
    public:
        AttributeAdapter(uint16_t& value)
            : ValueReference<uint16_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<uint16_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const int64_t& get() override;
        void set(const int64_t& value) override;
    };

    /// \brief Access a uint32_t as an int64_t
    template <>
    class NGRAPH_API AttributeAdapter<uint32_t> : public ValueReference<uint32_t>,
                                                  public ValueAccessor<int64_t>
    {
    public:
        AttributeAdapter(uint32_t& value)
            : ValueReference<uint32_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<uint32_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const int64_t& get() override;
        void set(const int64_t& value) override;
    };

    /// \brief Access a uint64_t as an int64_t
    template <>
    class NGRAPH_API AttributeAdapter<uint64_t> : public ValueReference<uint64_t>,
                                                  public ValueAccessor<int64_t>
    {
    public:
        AttributeAdapter(uint64_t& value)
            : ValueReference<uint64_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<uint64_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const int64_t& get() override;
        void set(const int64_t& value) override;
    };

#ifdef __APPLE__
    // size_t is one of the uint types on _WIN32
    template <>
    class NGRAPH_API AttributeAdapter<size_t> : public ValueReference<size_t>,
                                                public ValueAccessor<int64_t>
    {
    public:
        AttributeAdapter(size_t& value)
            : ValueReference<size_t>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<size_t>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const int64_t& get() override;
        void set(const int64_t& value) override;
    };
#endif

    /// Note: These class bodies cannot be defined with templates because of interactions
    /// between dllexport and templates on Windows.

    /// \brief Access a vector<int8_t>
    template <>
    class NGRAPH_API AttributeAdapter<std::vector<int8_t>>
        : public ValueReference<std::vector<int8_t>>, public ValueAccessor<std::vector<int8_t>>
    {
    public:
        AttributeAdapter(std::vector<int8_t>& value)
            : ValueReference<std::vector<int8_t>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<vector<int8_t>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const std::vector<int8_t>& get() override;
        void set(const std::vector<int8_t>& value) override;
    };

    /// \brief Access a vector<int16_t>
    template <>
    class NGRAPH_API AttributeAdapter<std::vector<int16_t>>
        : public ValueReference<std::vector<int16_t>>, public ValueAccessor<std::vector<int16_t>>
    {
    public:
        AttributeAdapter(std::vector<int16_t>& value)
            : ValueReference<std::vector<int16_t>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<vector<int16_t>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const std::vector<int16_t>& get() override;
        void set(const std::vector<int16_t>& value) override;
    };

    /// \brief Access a vector<int32_t>
    template <>
    class NGRAPH_API AttributeAdapter<std::vector<int32_t>>
        : public ValueReference<std::vector<int32_t>>, public ValueAccessor<std::vector<int32_t>>
    {
    public:
        AttributeAdapter(std::vector<int32_t>& value)
            : ValueReference<std::vector<int32_t>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<vector<int32_t>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const std::vector<int32_t>& get() override;
        void set(const std::vector<int32_t>& value) override;
    };

    /// \brief Access a vector<int64_t>
    template <>
    class NGRAPH_API AttributeAdapter<std::vector<int64_t>>
        : public ValueReference<std::vector<int64_t>>, public ValueAccessor<std::vector<int64_t>>
    {
    public:
        AttributeAdapter(std::vector<int64_t>& value)
            : ValueReference<std::vector<int64_t>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<vector<int64_t>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const std::vector<int64_t>& get() override;
        void set(const std::vector<int64_t>& value) override;
    };

    /// \brief Access a vector<uint8_t> as a vector<int8_t>
    template <>
    class NGRAPH_API AttributeAdapter<std::vector<uint8_t>>
        : public ValueReference<std::vector<uint8_t>>, public ValueAccessor<std::vector<int8_t>>
    {
    public:
        AttributeAdapter(std::vector<uint8_t>& value)
            : ValueReference<std::vector<uint8_t>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<vector<uint8_t>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const std::vector<int8_t>& get() override;
        void set(const std::vector<int8_t>& value) override;
    };

    /// \brief Access a vector<uint16_t> as a vector<int16_t>
    template <>
    class NGRAPH_API AttributeAdapter<std::vector<uint16_t>>
        : public ValueReference<std::vector<uint16_t>>, public ValueAccessor<std::vector<int16_t>>
    {
    public:
        AttributeAdapter(std::vector<uint16_t>& value)
            : ValueReference<std::vector<uint16_t>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<vector<uint16_t>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const std::vector<int16_t>& get() override;
        void set(const std::vector<int16_t>& value) override;
    };

    /// \brief Access a vector<uint32_t> as a vector<int32_t>
    template <>
    class NGRAPH_API AttributeAdapter<std::vector<uint32_t>>
        : public ValueReference<std::vector<uint32_t>>, public ValueAccessor<std::vector<int32_t>>
    {
    public:
        AttributeAdapter(std::vector<uint32_t>& value)
            : ValueReference<std::vector<uint32_t>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<vector<uint32_t>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const std::vector<int32_t>& get() override;
        void set(const std::vector<int32_t>& value) override;
    };

    /// \brief Access a vector<uint64_t> as a vector<int64_t>
    template <>
    class NGRAPH_API AttributeAdapter<std::vector<uint64_t>>
        : public ValueReference<std::vector<uint64_t>>, public ValueAccessor<std::vector<int64_t>>
    {
    public:
        AttributeAdapter(std::vector<uint64_t>& value)
            : ValueReference<std::vector<uint64_t>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<vector<uint64_t>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const std::vector<int64_t>& get() override;
        void set(const std::vector<int64_t>& value) override;
    };

#ifdef __APPLE__
    // size_t is not uint64_t on OSX
    template <>
    class NGRAPH_API AttributeAdapter<std::vector<size_t>>
        : public ValueReference<std::vector<size_t>>, public ValueAccessor<std::vector<int64_t>>
    {
    public:
        AttributeAdapter(std::vector<size_t>& value)
            : ValueReference<std::vector<size_t>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<vector<size_t>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const std::vector<int64_t>& get() override;
        void set(const std::vector<int64_t>& value) override;
    };
#endif

    /// \brief Access a vector<float>
    template <>
    class NGRAPH_API AttributeAdapter<std::vector<float>>
        : public ValueReference<std::vector<float>>, public ValueAccessor<std::vector<float>>
    {
    public:
        AttributeAdapter(std::vector<float>& value)
            : ValueReference<std::vector<float>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<vector<float>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const std::vector<float>& get() override;
        void set(const std::vector<float>& value) override;
    };

    /// \brief Access a vector<double>
    template <>
    class NGRAPH_API AttributeAdapter<std::vector<double>>
        : public ValueReference<std::vector<double>>, public ValueAccessor<std::vector<double>>
    {
    public:
        AttributeAdapter(std::vector<double>& value)
            : ValueReference<std::vector<double>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<vector<double>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const std::vector<double>& get() override;
        void set(const std::vector<double>& value) override;
    };

    /// \brief Access a vector<string>
    template <>
    class NGRAPH_API AttributeAdapter<std::vector<std::string>>
        : public ValueReference<std::vector<std::string>>,
          public ValueAccessor<std::vector<std::string>>
    {
    public:
        AttributeAdapter(std::vector<std::string>& value)
            : ValueReference<std::vector<std::string>>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<vector<string>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const std::vector<std::string>& get() override;
        void set(const std::vector<std::string>& value) override;
    };

    template <typename A, typename B>
    A copy_from(B& b)
    {
        A result(b.size());
        for (int i = 0; i < b.size(); ++i)
        {
            result[i] =
                static_cast<typename std::remove_reference<decltype(result[i])>::type>(b[i]);
        }
        return result;
    }
}
