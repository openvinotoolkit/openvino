// Copyright (C) 2018-2023 Intel Corporation
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
 * @ingroup ie_runtime_attr_api
 * @brief OldApiMapElementType class represents runtime info attribute that stores legacy type
 * that is required for obtaining IR in old API.
 */
class TRANSFORMATIONS_API ForceFP32 : public RuntimeAttribute {
public:
    OPENVINO_RTTI("force_fp32", "0");

    /**
     * A default constructor
     */
    ForceFP32() = default;

    /**
     * Constructs a new OldApiMapElementType object.
     * @param[in]  value  The object that stores values of OldApiMapElementType.
     */
    ForceFP32(const bool& value) : value(value) {}

    bool is_copyable() const override {
        return true;
    }

    bool visit_attributes(AttributeVisitor& visitor) override;

    bool value;
};

inline bool has_force_fp32(const std::shared_ptr<Node>& node) {
    const auto& rt_map = node->get_rt_info();
    return rt_map.count(ForceFP32::get_type_info_static());
}

inline ForceFP32 get_force_fp32(const std::shared_ptr<Node>& node) {
    const auto& rt_map = node->get_rt_info();
    return rt_map.at(ForceFP32::get_type_info_static()).as<ForceFP32>();
}

inline void set_force_fp32(const std::shared_ptr<Node>& node, const ForceFP32& old_api_map) {
    auto& rt_map = node->get_rt_info();
    rt_map[ForceFP32::get_type_info_static()] = old_api_map;
}

}  // namespace ov
