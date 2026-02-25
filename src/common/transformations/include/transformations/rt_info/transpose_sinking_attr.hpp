// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "transformations_visibility.hpp"

namespace ov {

TRANSFORMATIONS_API void mark_as_no_sinking_node(const std::shared_ptr<Node>& node);
TRANSFORMATIONS_API void reset_no_sinking_attribute(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API bool is_sinking_node(const std::shared_ptr<Node>& node);
TRANSFORMATIONS_API bool is_sinking_node(const Node* node);
TRANSFORMATIONS_API bool is_sinking_node(ov::Output<ov::Node> output);

/**
 * @ingroup ov_runtime_attr_api
 * @brief NoTransposeSinkingAttr class represents runtime info attribute that marks transpose
 * operation should not be moved be backward sinking propagation.
 */
class TRANSFORMATIONS_API NoTransposeSinkingAttr : public RuntimeAttribute {
public:
    OPENVINO_RTTI("no_transpose_sinking", "0", RuntimeAttribute);

    bool is_copyable() const override {
        return false;
    }
};

}  // namespace ov
