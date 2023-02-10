// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "transformations_visibility.hpp"

namespace ov {

TRANSFORMATIONS_API void mark_as_no_gather_sinking_node(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API bool is_gather_sinking_node(const std::shared_ptr<Node>& node);
TRANSFORMATIONS_API bool is_gather_sinking_node(const Node* node);
TRANSFORMATIONS_API bool is_gather_sinking_node(ov::Output<ov::Node> output);

/**
 * @ingroup ie_runtime_attr_api
 * @brief NoGatherSinkingAttr class represents runtime info attribute that marks gather
 * operation should not be moved be backward sinking propagation.
 */
class TRANSFORMATIONS_API NoGatherSinkingAttr : public RuntimeAttribute {
public:
    OPENVINO_RTTI("no_gather_sinking", "0");

    bool is_copyable() const override {
        return false;
    }
};

}  // namespace ov
