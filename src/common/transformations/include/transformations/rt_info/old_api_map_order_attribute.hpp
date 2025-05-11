// Copyright (C) 2018-2025 Intel Corporation
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
#include <set>
#include <string>
#include <utility>

#include "openvino/core/node.hpp"
#include "openvino/core/rtti.hpp"
#include "transformations_visibility.hpp"

namespace ov {

class OldApiMapOrder;
/**
 * @ingroup ov_runtime_attr_api
 * @brief OldApiMapOrder class represents runtime info attribute that stores
 * order of the transpose that is required for obtaining IR in old API.
 *
 *  OldApiMapOrder stores the following information.
 *  Parameter:
 *  Order of the transpose which should be applied to Parameter with old API layout to
 *  obtain Parameter with new API layout.
 *
 *  Result:
 *  Order of the transpose which should be applied to Result with new API layout to
 *  obtain Result with old API layout.
 */
class TRANSFORMATIONS_API OldApiMapOrder : public RuntimeAttribute {
public:
    OPENVINO_RTTI("old_api_map_order", "0", RuntimeAttribute);

    /**
     * A default constructor
     */
    OldApiMapOrder() = default;

    /**
     * Constructs a new OldApiMapOrder object.
     * @param[in]  value  The object that stores values of OldApiMapOrder.
     */
    OldApiMapOrder(const std::vector<uint64_t>& value) : value(value) {}

    bool is_copyable() const override {
        return false;
    }

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::vector<uint64_t> value;
};

inline bool has_old_api_map_order(const std::shared_ptr<Node>& node) {
    const auto& rt_map = node->get_rt_info();
    return rt_map.count(OldApiMapOrder::get_type_info_static());
}

inline OldApiMapOrder get_old_api_map_order(const std::shared_ptr<Node>& node) {
    const auto& rt_map = node->get_rt_info();
    return rt_map.at(OldApiMapOrder::get_type_info_static()).as<OldApiMapOrder>();
}

inline void set_old_api_map_order(std::shared_ptr<Node>& node, const OldApiMapOrder& old_api_map) {
    auto& rt_map = node->get_rt_info();
    rt_map[OldApiMapOrder::get_type_info_static()] = old_api_map;
}

}  // namespace ov
