// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <ie_core.hpp>
#include <ngraph_functions/builders.hpp>

#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "common_test_utils/common_utils.hpp"

#include "single_layer_tests/space_to_batch.hpp"

namespace LayerTestsDefinitions {

std::string SpaceToBatchLayerTest::getTestCaseName(const testing::TestParamInfo<spaceToBatchParamsTuple> &obj) {
    std::vector<size_t> inShapes, blockShape, padsBegin, padsEnd;
    InferenceEngine::Precision netPrc;
    std::string targetName;
    std::tie(blockShape, padsBegin, padsEnd, inShapes, netPrc, targetName) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inShapes) << "_";
    result << "netPRC=" << netPrc.name() << "_";
    result << "BS=" << CommonTestUtils::vec2str(blockShape) << "_";
    result << "PB=" << CommonTestUtils::vec2str(padsBegin) << "_";
    result << "PE=" << CommonTestUtils::vec2str(padsEnd) << "_";
    result << "targetDevice=" << targetName << "_";
    return result.str();
}

void SpaceToBatchLayerTest::SetUp() {
    SetRefMode(LayerTestsUtils::RefMode::INTERPRETER_TRANSFORMATIONS);
    std::vector<size_t> inputShape, blockShape, padsBegin, padsEnd;
    InferenceEngine::Precision inputPrecision, netPrecision;
    std::tie(blockShape, padsBegin, padsEnd, inputShape, netPrecision, targetDevice) = this->GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto s2b = ngraph::builder::makeSpaceToBatch(paramOuts[0], ngPrc, blockShape, padsBegin, padsEnd);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(s2b)};
    function = std::make_shared<ngraph::Function>(results, params, "SpaceToBatch");
}

TEST_P(SpaceToBatchLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions
