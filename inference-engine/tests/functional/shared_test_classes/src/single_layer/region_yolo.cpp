// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/region_yolo.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace LayerTestsDefinitions {
using namespace ov::test;

std::string RegionYoloLayerTest::getTestCaseName(const testing::TestParamInfo<regionYoloParamsTuple> &obj) {
    InputShape inputShape;
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
    InputShape inputShape;
    size_t classes;
    size_t coords;
    size_t num_regions;
    bool do_softmax;
    std::vector<int64_t> mask;
    int start_axis;
    int end_axis;
    InferenceEngine::Precision netPrecision;
    std::tie(inputShape, classes, coords, num_regions, do_softmax, mask, start_axis, end_axis, netPrecision, targetDevice) = this->GetParam();

    init_input_shapes({ inputShape });

    auto params = ngraph::builder::makeDynamicParams(inType, inputDynamicShapes);
    assert(params.size() == 1ul && "not expected params count");
    auto region_yolo = std::make_shared<ngraph::op::v0::RegionYolo>(params[0], coords, classes, num_regions, do_softmax, mask, start_axis, end_axis);
    function = std::make_shared<ngraph::Function>(std::make_shared<ngraph::opset1::Result>(region_yolo), params, "RegionYolo");
}

} // namespace LayerTestsDefinitions