// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional_test_utils/layer_test_utils/summary.hpp>
#include <ngraph_functions/subgraph_builders.hpp>

namespace ov {
namespace test {
namespace subgraph {


using OpGenerator = std::map<ngraph::NodeTypeInfo, std::function<std::shared_ptr<ov::Function>(const ov::DiscreteTypeInfo& typeInfo)>>;

OpGenerator getOpGeneratorMap();

static const std::vector<std::shared_ptr<ov::Function>> createFunctions() {
    auto opsets = LayerTestsUtils::Summary::getInstance().getOpSets();
    std::set<ngraph::NodeTypeInfo> opsInfo;
    for (const auto& opset : opsets) {
        const auto &type_info_set = opset.get_type_info_set();
        opsInfo.insert(type_info_set.begin(), type_info_set.end());
    }

    const std::vector<std::shared_ptr<ov::Function>> a = {
            ngraph::builder::subgraph::makeConvPoolRelu(),
            ngraph::builder::subgraph::makeConvPoolReluNonZero(),
    };
    return a;
}

}  // namespace subgraph
}  // namespace test
}  // namespace ov