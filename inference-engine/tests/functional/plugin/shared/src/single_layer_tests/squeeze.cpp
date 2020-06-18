// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <functional_test_utils/skip_tests_config.hpp>

#include "ie_precision.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "single_layer_tests/squeeze.hpp"

namespace LayerTestsDefinitions {
std::string SqueezeLayerTest::getTestCaseName(testing::TestParamInfo<squeezeParams> obj) {
    InferenceEngine::Precision netPrecision;
    SqueezeShape shapeItem;
    std::string targetDevice;
    bool isScalar;
    ngraph::helpers::SqueezeOpType opType;
    std::tie(shapeItem, opType, netPrecision, targetDevice, isScalar) = obj.param;

    std::ostringstream result;
    const char separator = '_';
    result << "OpType=" << opType << separator;
    result << "IS=" << CommonTestUtils::vec2str(shapeItem.first) << separator;
    result << "Axes=" << CommonTestUtils::vec2str(shapeItem.second) << separator;
    result << "netPRC=" << netPrecision.name() << separator;
    result << "targetDevice=" << targetDevice << separator;
    result << "isScalar=" << isScalar;
    return result.str();
}

std::vector<SqueezeShape> SqueezeLayerTest::combineShapes(const std::map<std::vector<size_t>, std::vector<std::vector<int>>>& inputShapes) {
    std::vector<SqueezeShape> resVec;
    for (auto& inputShape : inputShapes) {
        for (auto& item : inputShape.second) {
            resVec.push_back({inputShape.first, item});
        }
    }
    return resVec;
}

void SqueezeLayerTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::vector<size_t> inputShapes;
    std::vector<int> axesVector;
    SqueezeShape shapeItem;
    ngraph::helpers::SqueezeOpType opType;
    bool isScalar;
    std::tie(shapeItem, opType, netPrecision, targetDevice, isScalar) = GetParam();
    std::tie(inputShapes, axesVector) = shapeItem;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShapes});
    auto squeeze = ngraph::builder::makeSqueeze(params.front(), ngPrc, axesVector, opType);
    const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(squeeze)};
    function = std::make_shared<ngraph::Function>(results, params, "Squeeze");
}

TEST_P(SqueezeLayerTest, CompareWithRefs) {
    Run();
}

} // namespace LayerTestsDefinitions