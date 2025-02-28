// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "predicates.hpp"

namespace cldnn {

std::function<bool(const program_node& node)> not_in_shape_flow() {
    return [](const program_node& node) {
        return !node.is_in_shape_of_subgraph();
    };
}

std::function<bool(const program_node& node)> in_shape_flow() {
    return [](const program_node& node) {
        return node.is_in_shape_of_subgraph();
    };
}

}  // namespace cldnn
