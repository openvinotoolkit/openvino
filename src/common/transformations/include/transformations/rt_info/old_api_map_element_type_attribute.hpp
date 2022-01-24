// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines old API map attribute
 * @file old_api_map_element_type_attribute.hpp
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
/**
 * @ingroup ie_runtime_attr_api
 * @brief OldApiMapElementType class represents runtime info attribute that stores legacy type
 * that is required for obtaining IR in old API.
 */
class TRANSFORMATIONS_API OldApiMapElementType : public RuntimeAttribute {
public:
    OPENVINO_RTTI("old_api_map_element_type", "0");

    /**
     * A default constructor
     */
    OldApiMapElementType() = default;

    /**
     * Constructs a new OldApiMapElementType object.
     * @param[in]  value  The object that stores values of OldApiMapElementType.
     */
    OldApiMapElementType(const ngraph::element::Type& value) : value(value) {}

    bool is_copyable() const override {
        return false;
    }

    bool visit_attributes(AttributeVisitor& visitor) override;

    ngraph::element::Type value;
};

inline bool has_old_api_map_element_type(const std::shared_ptr<Node>& node) {
    const auto& rt_map = node->get_rt_info();
    return rt_map.count(OldApiMapElementType::get_type_info_static());
}

inline OldApiMapElementType get_old_api_map_element_type(const std::shared_ptr<Node>& node) {
    const auto& rt_map = node->get_rt_info();
    return rt_map.at(OldApiMapElementType::get_type_info_static()).as<OldApiMapElementType>();
}

inline void set_old_api_map_element_type(const std::shared_ptr<Node>& node, const OldApiMapElementType& old_api_map) {
    auto& rt_map = node->get_rt_info();
    rt_map[OldApiMapElementType::get_type_info_static()] = old_api_map;
}

}  // namespace ov