// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/node.hpp"
#include "ngraph/type.hpp"

namespace ngraph
{
    using VariantTypeInfo = DiscreteTypeInfo;

    class NGRAPH_API Variant
    {
    public:
        virtual ~Variant();
        virtual const VariantTypeInfo& get_type_info() const = 0;

        virtual std::shared_ptr<ngraph::Variant> init(const std::shared_ptr<ngraph::Node>& node);
        virtual std::shared_ptr<ngraph::Variant> merge(const ngraph::NodeVector& nodes);
        virtual std::string to_string() { return ""; }
    };

    template <typename VT>
    class VariantImpl : public Variant
    {
    public:
        using value_type = VT;

        VariantImpl(const value_type& value)
            : m_value(value)
        {
        }

        const value_type& get() const { return m_value; }
        value_type& get() { return m_value; }
        void set(const value_type& value) { m_value = value; }

    protected:
        value_type m_value;
    };

    extern template class NGRAPH_API VariantImpl<std::string>;
    extern template class NGRAPH_API VariantImpl<int64_t>;

    template <typename VT>
    class VariantWrapper
    {
    };

    template <>
    class NGRAPH_API VariantWrapper<std::string> : public VariantImpl<std::string>
    {
    public:
        static constexpr VariantTypeInfo type_info{"Variant::std::string", 0};
        const VariantTypeInfo& get_type_info() const override { return type_info; }
        VariantWrapper(const value_type& value)
            : VariantImpl<value_type>(value)
        {
        }
    };

    template <>
    class NGRAPH_API VariantWrapper<int64_t> : public VariantImpl<int64_t>
    {
    public:
        static constexpr VariantTypeInfo type_info{"Variant::int64_t", 0};
        const VariantTypeInfo& get_type_info() const override { return type_info; }
        VariantWrapper(const value_type& value)
            : VariantImpl<value_type>(value)
        {
        }
    };

    template <typename T>
    inline std::shared_ptr<Variant> make_variant(const T& p)
    {
        return std::dynamic_pointer_cast<VariantImpl<T>>(std::make_shared<VariantWrapper<T>>(p));
    }

    template <size_t N>
    inline std::shared_ptr<Variant> make_variant(const char (&s)[N])
    {
        return std::dynamic_pointer_cast<VariantImpl<std::string>>(
            std::make_shared<VariantWrapper<std::string>>(s));
    }

} // namespace ngraph
