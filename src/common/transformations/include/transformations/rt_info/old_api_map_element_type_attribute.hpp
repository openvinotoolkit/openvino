// Copyright (C) 2018-2025 Intel Corporation
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
#include <set>
#include <string>
#include <utility>

#include "openvino/core/node.hpp"
#include "openvino/core/rtti.hpp"
#include "transformations_visibility.hpp"

namespace ov {
/**
 * @ingroup ov_runtime_attr_api
 * @brief OldApiMapElementType class represents runtime info attribute that stores legacy type
 * that is required for obtaining IR in old API.
 */
class TRANSFORMATIONS_API OldApiMapElementType : public RuntimeAttribute {
public:
    OPENVINO_RTTI("old_api_map_element_type", "0", RuntimeAttribute);

    /**
     * A default constructor
     */
    OldApiMapElementType() = default;

    /**
     * Constructs a new OldApiMapElementType object.
     * @param[in]  value  The object that stores values of OldApiMapElementType.
     */
    OldApiMapElementType(const ov::element::Type& value) : value(value) {}

    bool is_copyable() const override {
        return false;
    }

    bool visit_attributes(AttributeVisitor& visitor) override;

    ov::element::Type value;
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
