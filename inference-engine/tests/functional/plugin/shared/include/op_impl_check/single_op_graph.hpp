// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional_test_utils/layer_test_utils/summary.hpp>
#include <ngraph_functions/subgraph_builders.hpp>

namespace ov {
namespace test {
namespace subgraph {

static std::vector<std::shared_ptr<ov::Function>> createFunctions() {
    auto opsets = LayerTestsUtils::Summary::getInstance().getOpSets();
    for (const auto& opset : opsets) {
        std::cout << opset.size();
    }
    const std::vector<std::shared_ptr<ov::Function>> a = {
            ngraph::builder::subgraph::makeConvPoolRelu(),
            ngraph::builder::subgraph::makeConvPoolReluNonZero(),
            nullptr
    };
    return a;
}

}  // namespace subgraph
}  // namespace test
}  // namespace ov