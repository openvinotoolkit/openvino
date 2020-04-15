// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/nonzero.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "ie_core.hpp"

#include <tuple>
#include <string>
#include <vector>
#include <memory>

namespace LayerTestsDefinitions {

std::string NonZeroLayerTest::getTestCaseName(testing::TestParamInfo<NonZeroLayerTestParamsSet> obj) {
    std::vector<size_t> inputShape;
    InferenceEngine::Precision inputPrecision, netPrecision;
    std::string targetDevice;
    ConfigMap config;
    std::tie(inputShape, inputPrecision, netPrecision, targetDevice, config) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "inPRC=" << inputPrecision.name() << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void NonZeroLayerTest::SetUp() {
    std::vector<size_t> inputShape;
    std::tie(inputShape, inputPrecision, netPrecision, targetDevice, config) = this->GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto paramNode = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape(inputShape));

    auto nonZeroOp = std::make_shared<ngraph::opset3::NonZero>(paramNode->output(0));

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(nonZeroOp)};
    fnPtr = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{paramNode}, "non_zero");
}

TEST_P(NonZeroLayerTest, CompareWithRefs) {
    inferAndValidate();
}
}  // namespace LayerTestsDefinitions
