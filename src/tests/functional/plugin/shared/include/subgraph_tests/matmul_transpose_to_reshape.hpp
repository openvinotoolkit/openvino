// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/matmul_transpose_to_reshape.hpp"

namespace ov {
namespace test {

TEST_P(MatMulTransposeToReshape, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();

    const auto runtime_model = compiledModel.get_runtime_model();
    ASSERT_NE(runtime_model, nullptr);

    bool has_fc = false;
    for (const auto& node : runtime_model->get_ordered_ops()) {
        const auto& rt_info = node->get_rt_info();
        const auto it = rt_info.find("layerType");
        if (it == rt_info.end()) {
            continue;
        }

        const auto layer_type = it->second.as<std::string>();
        if (layer_type == "FullyConnected") {
            has_fc = true;
        }
        ASSERT_NE(layer_type, "Transpose");
        ASSERT_NE(layer_type, "Permute");
    }

    ASSERT_TRUE(has_fc);
}

}  // namespace test
}  // namespace ov
