// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/reorg_yolo.hpp"

#include "common_test_utils/common_utils.hpp"
#include "ngraph_functions/builders.hpp"

namespace ov {
namespace test {
namespace subgraph {

std::string ReorgYoloLayerTest::getTestCaseName(const testing::TestParamInfo<ReorgYoloParamsTuple>& obj) {
    InputShape inputShape;
    size_t stride;
    ElementType netPrecision;
    TargetDevice targetDev;
    std::tie(inputShape, stride, netPrecision, targetDev) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::partialShape2str({inputShape.first}) << "_";
    for (const auto& item : inputShape.second) {
        result << CommonTestUtils::vec2str(item) << "_";
    }
    result << "stride=" << stride << "_";
    result << "netPRC=" << netPrecision << "_";
    result << "targetDevice=" << targetDev << "_";
    return result.str();
}

void ReorgYoloLayerTest::SetUp() {
    InputShape inputShape;
    size_t stride;
    ElementType netPrecision;
    std::tie(inputShape, stride, netPrecision, targetDevice) = this->GetParam();

    init_input_shapes({inputShape});

    auto param = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, inputDynamicShapes[0]);
    auto reorg_yolo = std::make_shared<ngraph::op::v0::ReorgYolo>(param, stride);
    function = std::make_shared<ngraph::Function>(std::make_shared<ngraph::opset1::Result>(reorg_yolo),
                                                  ngraph::ParameterVector{param},
                                                  "ReorgYolo");
}

}  // namespace subgraph
}  // namespace test
}  // namespace ov
