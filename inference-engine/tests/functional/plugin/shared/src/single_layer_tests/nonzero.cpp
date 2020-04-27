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
    InferenceEngine::Precision inputPrecision;
    std::string targetDevice;
    std::tie(inputShape, inputPrecision, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "inPRC=" << inputPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void NonZeroLayerTest::SetUp() {
    SetRefMode(LayerTestsUtils::RefMode::CONSTANT_FOLDING);
    auto inputShape     = std::vector<std::size_t>{};
    auto inputPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(inputShape, inputPrecision, targetDevice) = GetParam();

    const auto& precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
    const auto& paramNode = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));

    auto nonZeroOp = std::make_shared<ngraph::opset3::NonZero>(paramNode->output(0));

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(nonZeroOp)};
    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{paramNode}, "non_zero");
}

TEST_P(NonZeroLayerTest, CompareWithReference) {
    Run();

    if (targetDevice == std::string{CommonTestUtils::DEVICE_GPU}) {
        PluginCache::get().reset();
    }
}
}  // namespace LayerTestsDefinitions
