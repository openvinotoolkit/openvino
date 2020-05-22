// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/empty_graph.hpp"

namespace LayerTestsDefinitions {

std::string EmptyGraph::getTestCaseName(testing::TestParamInfo<LayerTestsUtils::basicParams> obj) {
    return LayerTestsUtils::getTestCaseName(obj);
}

void EmptyGraph::SetUp() {
    std::vector<size_t> inputShape;
    std::tie(inPrc, inputShape, targetDevice) = this->GetParam();
    outPrc = inPrc;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(params[0])};
    function = std::make_shared<ngraph::Function>(results, params, "EmptyGraph");
}

TEST_P(EmptyGraph, CompareWithRefImpl) {
    Run();
};

} // namespace LayerTestsDefinitions
