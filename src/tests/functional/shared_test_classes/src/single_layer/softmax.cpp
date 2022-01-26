// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

#include "common_test_utils/common_utils.hpp"

#include "shared_test_classes/single_layer/softmax.hpp"

namespace ov {
namespace test {
namespace subgraph {

std::string SoftMaxLayerTest::getTestCaseName(const testing::TestParamInfo<SoftMaxTestParams>& obj) {
    ElementType netType, inType, outType;
    InputShape shapes;
    size_t axis;
    TargetDevice targetDevice;
    Config config;
    std::tie(netType, inType, outType, shapes, axis, targetDevice, config) = obj.param;

    std::ostringstream result;
    result << "NetType=" << netType << "_";
    result << "InType=" << inType << "_";
    result << "OutType=" << outType << "_";
    result << "IS=" << CommonTestUtils::partialShape2str({shapes.first}) << "_";
    result << "TS=";
    for (const auto& item : shapes.second) {
        result << CommonTestUtils::vec2str(item) << "_";
    }
    result << "Axis=" << axis << "_";
    result << "Device=" << targetDevice;

    return result.str();
}

void SoftMaxLayerTest::SetUp() {
    InputShape shapes;
    ElementType ngPrc;
    size_t axis;

    std::tie(ngPrc, inType, outType, shapes, axis, targetDevice, configuration) = GetParam();
    init_input_shapes({shapes});

    const auto params = ngraph::builder::makeDynamicParams(ngPrc, inputDynamicShapes);
    const auto paramOuts =
            ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    const auto softMax = std::make_shared<ngraph::opset1::Softmax>(paramOuts.at(0), axis);
    const ngraph::ResultVector results {std::make_shared<ngraph::opset1::Result>(softMax)};

    function = std::make_shared<ngraph::Function>(results, params, "softMax");
}
}  // namespace subgraph
}  // namespace test
}  // namespace ov