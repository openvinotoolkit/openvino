// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_core.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "single_layer_tests/region_yolo.hpp"

namespace LayerTestsDefinitions {

std::string RegionYoloLayerTest::getTestCaseName(const testing::TestParamInfo<regionYoloParamsTuple> &obj) {
    ngraph::Shape inputShape;
    size_t classes;
    size_t coords;
    size_t num_regions;
    bool do_softmax;
    std::vector<int64_t> mask;
    int start_axis;
    int end_axis;
    InferenceEngine::Precision netPrecision;
    std::string targetName;
    std::tie(inputShape, classes, coords, num_regions, do_softmax , mask, start_axis, end_axis, netPrecision, targetName) = obj.param;
    std::ostringstream result;
    result << "IS=" << inputShape << "_";
    result << "classes=" << classes << "_";
    result << "coords=" << coords << "_";
    result << "num=" << num_regions << "_";
    result << "doSoftmax=" << do_softmax << "_";
    result << "axis=" << start_axis << "_";
    result << "endAxis=" << end_axis << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetName << "_";
    return result.str();
}

void RegionYoloLayerTest::SetUp() {
    ngraph::Shape inputShape;
    size_t classes;
    size_t coords;
    size_t num_regions;
    bool do_softmax;
    std::vector<int64_t> mask;
    int start_axis;
    int end_axis;
    InferenceEngine::Precision netPrecision;
    std::tie(inputShape, classes, coords, num_regions, do_softmax, mask, start_axis, end_axis, netPrecision, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto param = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, inputShape);
    auto region_yolo = std::make_shared<ngraph::op::v0::RegionYolo>(param, coords, classes, num_regions, do_softmax, mask, start_axis, end_axis);
    function = std::make_shared<ngraph::Function>(std::make_shared<ngraph::opset1::Result>(region_yolo), ngraph::ParameterVector{param}, "RegionYolo");
}

TEST_P(RegionYoloLayerTest, CompareWithRefs) {
    Run();
};

} // namespace LayerTestsDefinitions