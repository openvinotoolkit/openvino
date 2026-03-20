// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/gated_delta_net.hpp"

namespace ov::test {

inline void CheckNumberOfNodesWithType(std::shared_ptr<const ov::Model> function,
                                       const std::unordered_set<std::string>& nodeTypes,
                                       size_t expectedCount) {
    ASSERT_NE(nullptr, function);
    int num_ops = 0;
    for (const auto& node : function->get_ordered_ops()) {
        const auto& rt_info = node->get_rt_info();
        const auto layer_type = rt_info.find("layerType")->second.as<std::string>();
        if (nodeTypes.count(layer_type)) {
            num_ops++;
        }
    }
    ASSERT_EQ(num_ops, expectedCount);
}

TEST_P(GatedDeltaNet, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    auto function = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithType(function, {"GatedDeltaNet"}, 1);
    CheckNumberOfNodesWithType(function, {"Transpose"}, 0);
    CheckNumberOfNodesWithType(function, {"Concat"}, 0);
    CheckNumberOfNodesWithType(function, {"ReduceSum"}, 0);
    CheckNumberOfNodesWithType(function, {"Multiply"}, 0);
    CheckNumberOfNodesWithType(function, {"Divide"}, 0);
};
}  // namespace ov::test