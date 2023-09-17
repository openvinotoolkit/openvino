// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"

namespace ov {
namespace intel_gna {
namespace rt_info {

void set_node_id(const std::shared_ptr<Node>& node, uint64_t id);

void remove_node_id(const std::shared_ptr<Node>& node);

uint64_t get_node_id(const std::shared_ptr<Node>& node);

/**
 * @ingroup ie_runtime_attr_api
 * @brief GnaNodeId class represents runtime info attribute that marks operation
 * with order id
 */
class GnaNodeId : public RuntimeAttribute {
public:
    OPENVINO_RTTI("gna_node_id", "0");

    GnaNodeId() = default;

    bool is_copyable() const override {
        return false;
    }
};
}  // namespace rt_info
}  // namespace intel_gna
}  // namespace ov
