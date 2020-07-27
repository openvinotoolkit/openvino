// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <functional_test_utils/skip_tests_config.hpp>
#include <ngraph/ops.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "single_layer_tests/activation_mish.hpp"

namespace LayerTestsDefinitions {

std::string MishLayerTest::getTestCaseName(testing::TestParamInfo<mishLayerTestParamsSet> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(netPrecision, inputShapes, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void MishLayerTest::SetUp() {
    std::vector<size_t> inputShape;
    auto netPrecision   = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(netPrecision, inputShape, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto mish = std::dynamic_pointer_cast<ngraph::op::v4::Mish>(
            ngraph::builder::makeActivationMish(paramOuts[0]));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(mish)};
    function = std::make_shared<ngraph::Function>(results, params, "mish");
}

TEST_P(MishLayerTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions
