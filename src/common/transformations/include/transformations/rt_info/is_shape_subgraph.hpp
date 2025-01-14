// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "transformations_visibility.hpp"

namespace ov {

TRANSFORMATIONS_API void mark_shape_subgraph(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API void unmark_shape_subgraph(const std::shared_ptr<Node>& node);

TRANSFORMATIONS_API bool is_shape_subgraph(const std::shared_ptr<const Node>& node);

/**
 * @ingroup ov_runtime_attr_api
 * @brief ShapeSubgraph class represents runtime info attribute that marks shape subgraphs.
 * Information whether the node belongs to the shape path or to the data path is needed during evaluate and CF.
 */
class TRANSFORMATIONS_API ShapeSubgraph : public RuntimeAttribute {
public:
    OPENVINO_RTTI("shape_subgraph", "0", RuntimeAttribute);

    ShapeSubgraph() = default;

    bool is_copyable() const override {
        return false;
    }
};

}  // namespace ov
