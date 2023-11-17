// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/tile.hpp"

namespace LayerTestsDefinitions {

std::string TileLayerTest::getTestCaseName(const testing::TestParamInfo<TileLayerTestParamsSet>& obj) {
    TileSpecificParams tileParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(tileParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "Repeats=" << ov::test::utils::vec2str(tileParams) << "_";
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
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

    auto repeatsNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, std::vector<size_t>{tileParams.size()}, tileParams);
    auto tile = std::make_shared<ov::op::v0::Tile>(params[0], repeatsNode);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(tile)};
    function = std::make_shared<ngraph::Function>(results, params, "tile");
}

}  // namespace LayerTestsDefinitions
