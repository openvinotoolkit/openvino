// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
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
    };

    template <typename VT>
    class VariantImpl : public Variant
    {
    public:
        using value_type = VT;

        template <typename... Args>
        VariantImpl(Args&&... args)
            : m_value{std::forward<Args>(args)...}
        {
        }

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

    template <typename VT>
    class VariantWrapper
    {
    };

    template <typename T, typename... Args>
    std::shared_ptr<ngraph::VariantWrapper<T>> make_variant(Args&&... args)
    {
        return std::make_shared<ngraph::VariantWrapper<T>>(std::forward<Args>(args)...);
    }

    template <typename T, typename>
    T& variant_cast(const std::shared_ptr<Variant>& variant)
    {
        auto variant_wrapper_ptr = ngraph::as_type_ptr<ngraph::VariantWrapper<T>>(variant);
        NGRAPH_CHECK(variant_wrapper_ptr != nullptr);
        return variant_wrapper_ptr->get();
    }

#define NGRAPH_VARIANT_DEFINITION(TYPE, TYPE_NAME, VERSION_INDEX)                                  \
    template class ngraph::VariantImpl<TYPE>;                                                      \
    NGRAPH_RTTI_DEFINITION(VariantWrapper<TYPE>, TYPE_NAME, VERSION_INDEX);

#define NGRAPH_VARIANT_DECLARATION(TYPE)                                                           \
    extern template class NGRAPH_API VariantImpl<TYPE>;                                            \
    template <>                                                                                    \
    class NGRAPH_API VariantWrapper<TYPE> : public VariantImpl<TYPE>                               \
    {                                                                                              \
    public:                                                                                        \
        NGRAPH_RTTI_DECLARATION;                                                                   \
        VariantWrapper(const VariantWrapper<TYPE>&) = default;                                     \
        VariantWrapper(VariantWrapper<TYPE>&&) = default;                                          \
        using VariantImpl<TYPE>::VariantImpl;                                                      \
    };

    NGRAPH_VARIANT_DECLARATION(std::string);
    NGRAPH_VARIANT_DECLARATION(std::int64_t);
} // namespace ngraph
