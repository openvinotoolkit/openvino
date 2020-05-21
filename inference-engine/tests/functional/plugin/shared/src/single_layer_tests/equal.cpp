// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/equal.hpp"

#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <ie_core.hpp>


namespace LayerTestsDefinitions {

std::string EqualLayerTest::getTestCaseName(const testing::TestParamInfo<EqualTestParam>& obj) {
    InferenceEngine::Precision inPrecision;
    InferenceEngine::Precision outPrecision;
    std::vector<InferenceEngine::SizeVector> inputShapes;
    std::string targetDevice;

    std::tie(inputShapes, inPrecision, outPrecision, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "inPrc=" << inPrecision.name() << "_";
    result << "outPrc=" << outPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;

    return result.str();
}

void EqualLayerTest::SetUp() {
    std::vector<InferenceEngine::SizeVector> inputShapes;
    InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::UNSPECIFIED;

    std::tie(inputShapes, inputPrecision, outPrc, targetDevice) = this->GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
    auto paramsVector = ngraph::builder::makeParams(ngPrc, {inputShapes});
    IE_ASSERT(paramsVector.size() == 2);

    auto equalOp = std::make_shared<ngraph::opset3::Equal>(paramsVector[0], paramsVector[1]);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(equalOp)};

    function = std::make_shared<ngraph::Function>(results, paramsVector, "Equal");
}

TEST_P(EqualLayerTest, CompareWithRefs) {
    Run();

    if (targetDevice == std::string{CommonTestUtils::DEVICE_GPU}) {
        PluginCache::get().reset();
    }
}
}  // namespace LayerTestsDefinitions
