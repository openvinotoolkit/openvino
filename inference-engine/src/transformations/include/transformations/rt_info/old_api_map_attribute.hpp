// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines old API map attribute
 * @file old_api_map_attribute.hpp
 */

#pragma once

#include <assert.h>

#include <functional>
#include <memory>
#include <ngraph/attribute_visitor.hpp>
#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <openvino/core/rtti.hpp>
#include <set>
#include <string>
#include <transformations_visibility.hpp>
#include <utility>

namespace ov {

class OldApiMap;
/**
 * @ingroup ie_runtime_attr_api
 * @brief OldApiMapAttr class stores the value of OldApiMap class.
 */
class TRANSFORMATIONS_API OldApiMapAttr {
private:
    std::vector<uint64_t> m_order;
    ngraph::element::Type m_type;

public:
    friend class OldApiMap;

    /**
     * A default constructor
     */
    OldApiMapAttr() = default;

    /**
     * @brief      Constructs a new OldApiMapAttr object.
     * @param[in]  order  Transpose order.
     * @param[in]  type  Legacy type.
     */
    explicit OldApiMapAttr(std::vector<uint64_t> order, ngraph::element::Type type) {
        m_order = std::move(order);
        m_type = type;
    }

    /**
     * @brief Returns the transpose order that should be used for obtain a node with old API layout.
     * @return transpose order.
     */
    std::vector<uint64_t> get_order() const {
        return m_order;
    }

    /**
     * @brief Returns the legacy type of the node.
     * @return legacy type.
     */
    ngraph::element::Type get_type() const {
        return m_type;
    }
};

/**
 * @ingroup ie_runtime_attr_api
 * @brief OldApiMap class represents runtime info attribute that stores legacy type
 * and order of the transpose that is required for obtaining IR in old API.
 */
class TRANSFORMATIONS_API OldApiMap : public VariantImpl<OldApiMapAttr> {
public:
    OPENVINO_RTTI("old_api_map", "0");

    /**
     * A default constructor
     */
    OldApiMap() = default;

    /**
     * Constructs a new OldApiMap object.
     * @param[in]  value  The object that stores values of OldApiMap.
     */
    OldApiMap(const value_type& value) : VariantImpl<value_type>(value) {}

    bool is_copyable() const override {
        return false;
    }

    bool visit_attributes(AttributeVisitor& visitor) override;
};

}  // namespace ov
