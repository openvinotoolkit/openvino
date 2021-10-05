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
#include <string>
#include <vector>
#include <set>

#include <ngraph/node.hpp>
#include <utility>
#include <ngraph/variant.hpp>
#include <openvino/core/rtti.hpp>
#include <ngraph/attribute_visitor.hpp>
#include <transformations_visibility.hpp>

namespace ov {

class OldApiMap;
/**
 * @ingroup ie_runtime_attr_api
 * @brief OldApiMap class represents runtime info attribute that stores legacy type
 * and order of the transpose that is required for obtaining IR in old API.
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

    explicit OldApiMapAttr(std::vector<uint64_t> order, ngraph::element::Type type) {
        m_order = std::move(order);
        m_type = type;
    }

    std::vector<uint64_t> get_order() const {
        return m_order;
    }

    ngraph::element::Type get_type() const {
        return m_type;
    }
};

class TRANSFORMATIONS_API OldApiMap : public VariantImpl<OldApiMapAttr> {
public:
    OPENVINO_RTTI("old_api_map", "0");

    OldApiMap() = default;

    OldApiMap(const value_type &value) : VariantImpl<value_type>(value) {}

    bool is_copyable() const override { return false; }

    bool visit_attributes(AttributeVisitor & visitor) override;
};

}  // namespace ov
