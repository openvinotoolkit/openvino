// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "transformations_visibility.hpp"

namespace ov {

TRANSFORMATIONS_API void mark_as_dequantization_node(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API bool is_dequantization_node(const std::shared_ptr<const Node>& node);

TRANSFORMATIONS_API void unmark_dequantization_node(const std::shared_ptr<Node>& node);

/**
 * @ingroup ov_runtime_attr_api
 * @brief DequantizationNode class represents runtime info attribute that marks operation
 * that are part of dequantization subgraph.
 */
class TRANSFORMATIONS_API DequantizationNode : public RuntimeAttribute {
public:
    OPENVINO_RTTI("dequantization_node", "0", RuntimeAttribute);

    bool is_copyable() const override {
        return false;
    }
};

}  // namespace ov
