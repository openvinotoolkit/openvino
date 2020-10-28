// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines primitives priority attribute
 * @file primitives_priority_attribute.hpp
 */

#pragma once

#include <assert.h>
#include <functional>
#include <memory>
#include <string>
#include <set>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {

/**
 * @ingroup ie_runtime_attr_api
 * @brief PrimitivesPriority class represents runtime info attribute that
 * can be used for plugins specific primitive choice.
 */
class TRANSFORMATIONS_API PrimitivesPriority {
private:
    std::string primitives_priority;

public:
    /**
     * A default constructor
     */
    PrimitivesPriority() = default;

    /**
     * @brief      Constructs a new object consisting of a single name     *
     * @param[in]  name  The primitives priority value
     */
    explicit PrimitivesPriority(const std::string &primitives_priority) : primitives_priority(primitives_priority) {}

    /**
     * @brief return string with primitives priority value
     */
    std::string getPrimitivesPriority() const;
};

extern template class TRANSFORMATIONS_API VariantImpl<PrimitivesPriority>;

template<>
class TRANSFORMATIONS_API VariantWrapper<PrimitivesPriority> : public VariantImpl<PrimitivesPriority> {
public:
    static constexpr VariantTypeInfo type_info{"Variant::RuntimeAttribute::PrimitivesPriority", 0};

    const VariantTypeInfo &get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type &value) : VariantImpl<value_type>(value) {}

    std::shared_ptr<ngraph::Variant> merge(const ngraph::NodeVector & nodes) override;

    std::shared_ptr<ngraph::Variant> init(const std::shared_ptr<ngraph::Node> & node) override;
};

/**
 * @ingroup ie_runtime_attr_api
 * @brief getPrimitivesPriority return string with primitive priorities value
 * @param[in] node The node will be used to get PrimitivesPriority attribute
 */
TRANSFORMATIONS_API std::string getPrimitivesPriority(const std::shared_ptr<ngraph::Node> & node);

}  // namespace ngraph
