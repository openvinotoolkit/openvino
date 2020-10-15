// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_core.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "single_layer_tests/reorg_yolo.hpp"

namespace LayerTestsDefinitions {

std::string ReorgYoloLayerTest::getTestCaseName(const testing::TestParamInfo<ReorgYoloParamsTuple> &obj) {
    ngraph::Shape inputShape;
    size_t stride;
    InferenceEngine::Precision netPrecision;
    std::string targetName;
    std::tie(inputShape, stride, netPrecision, targetName) = obj.param;
    std::ostringstream result;
    result << "IS=" << inputShape << "_";
    result << "stride=" << stride << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetName << "_";
    return result.str();
}

void ReorgYoloLayerTest::SetUp() {
    ngraph::Shape inputShape;
    size_t stride;
    InferenceEngine::Precision netPrecision;
    std::tie(inputShape, stride, netPrecision, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto param = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, inputShape);
    auto reorg_yolo = std::make_shared<ngraph::op::v0::ReorgYolo>(param, stride);
    function = std::make_shared<ngraph::Function>(std::make_shared<ngraph::opset1::Result>(reorg_yolo), ngraph::ParameterVector{param}, "ReorgYolo");
}

TEST_P(ReorgYoloLayerTest, CompareWithRefs) {
    Run();
};

} // namespace LayerTestsDefinitions
