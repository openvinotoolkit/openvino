// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/msda_pattern.hpp"

namespace ov {
namespace test {

TEST_F(MSDAPattern, run) {
    run();
    // The StridedSlice slices are grouped into a VariadicSplit by the GPU
    // pipeline, and the pattern must be fused into a single MSDA primitive.
    size_t msda_nodes = 0;
    for (const auto& op : compiledModel.get_runtime_model()->get_ops()) {
        const auto rt = op->get_rt_info();
        const auto it = rt.find("layerType");
        if (it != rt.end() && it->second.as<std::string>().find("msda") != std::string::npos)
            ++msda_nodes;
    }
    EXPECT_EQ(msda_nodes, 1u);
}

}  // namespace test
}  // namespace ov