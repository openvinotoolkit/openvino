// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/normalize_l2.hpp"


namespace LayerTestsDefinitions {

std::string NormalizeL2LayerTest::getTestCaseName(testing::TestParamInfo<NormalizeL2LayerTestParams> obj) {
    std::vector<int64_t> axes;
    float eps;
    ngraph::op::EpsMode epsMode;
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::tie(axes, eps, epsMode, inputShape, netPrecision, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "axes=" << CommonTestUtils::vec2str(axes) << "_";
    result << "eps=" << eps << "_";
    result << "epsMode=" << epsMode << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void NormalizeL2LayerTest::SetUp() {
    InferenceEngine::SizeVector inputShape;
    std::vector<int64_t> axes;
    float eps;
    ngraph::op::EpsMode epsMode;
    InferenceEngine::Precision netPrecision;
    std::tie(axes, eps, epsMode, inputShape, netPrecision, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto norm = ngraph::builder::makeNormalizeL2(params[0], axes, eps, epsMode);
    ngraph::ResultVector results{std::make_shared<ngraph::opset4::Result>(norm)};
    function = std::make_shared<ngraph::Function>(results, params, "NormalizeL2");
}

TEST_P(NormalizeL2LayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions
