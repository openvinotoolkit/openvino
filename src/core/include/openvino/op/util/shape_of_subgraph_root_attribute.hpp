// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"

namespace ov {

void OPENVINO_API mark_as_shape_of_subgraph_root(const std::shared_ptr<Node>& node);

bool OPENVINO_API is_shape_of_subgraph_root(const std::shared_ptr<const Node>& node);

/**
 * @brief ShapeOfSubgraphRoot marks a node whose output is a shape/index-style scalar (not real
 * tensor data), e.g. a data-dependent begin/end/index value feeding Slice, Gather or ScatterUpdate.
 * Plugins that already give literal ShapeOf-derived subgraphs a correctness-focused execution path
 * (host-computed, protected from elementwise fusion) can extend that same protection to this node
 * and everything computed from it.
 */
class OPENVINO_API ShapeOfSubgraphRoot : public RuntimeAttribute {
public:
    OPENVINO_RTTI("shape_of_subgraph_root", "0", RuntimeAttribute);

    ShapeOfSubgraphRoot() = default;

    bool is_copyable() const override {
        return false;
    }
};

}  // namespace ov
