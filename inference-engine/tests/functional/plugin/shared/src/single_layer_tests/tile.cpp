// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <functional>

#include "single_layer_tests/tile.hpp"


namespace LayerTestsDefinitions {

std::string TileLayerTest::getTestCaseName(testing::TestParamInfo<TileLayerTestParamsSet> obj) {
    TileSpecificParams tileParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(tileParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "Repeats=" << CommonTestUtils::vec2str(tileParams) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void TileLayerTest::SetUp() {
    TileSpecificParams tileParams;
    std::vector<size_t> inputShape;
    auto netPrecision   = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(tileParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto tile = ngraph::builder::makeTile(paramOuts[0], tileParams);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(tile)};
    function = std::make_shared<ngraph::Function>(results, params, "tile");
}

TEST_P(TileLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions
