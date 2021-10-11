// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/softmax.hpp"

namespace LayerTestsDefinitions {

std::string SoftMaxLayerTest::getTestCaseName(const testing::TestParamInfo<softMaxLayerTestParams>& obj) {
    ngraph::element::Type_t netPrecision;
//    InferenceEngine::Precision inPrc, outPrc;
    std::pair<ngraph::PartialShape, std::vector<ngraph::Shape>> shapes;
    size_t axis;
    std::string targetDevice;
    std::map<std::string, std::string> config;
    std::tie(netPrecision, shapes, axis, targetDevice, config) = obj.param;

    std::ostringstream result;
    result << "netPRC=" << netPrecision << "_";
//    result << "inPRC=" << inPrc.name() << "_";
//    result << "outPRC=" << outPrc.name() << "_";
    result << "IS=" << CommonTestUtils::partialShape2str({shapes.first}) << "_";
    result << "TS=";
    for (const auto& item : shapes.second) {
        result << CommonTestUtils::vec2str(item) << "_";
    }
    result << "axis=" << axis << "_";
    result << "trgDev=" << targetDevice;

    return result.str();
}

void SoftMaxLayerTest::SetUp() {
    std::pair<ov::PartialShape, std::vector<ov::Shape>> shapes;
    ngraph::element::Type_t ngPrc;
    size_t axis;

    std::tie(ngPrc, shapes, axis, targetDevice, configuration) = GetParam();
    init_input_shapes(shapes);

    const auto params = ngraph::builder::makeDynamicParams(ngPrc, inputDynamicShapes);
    const auto paramOuts =
            ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    const auto softMax = std::make_shared<ngraph::opset1::Softmax>(paramOuts.at(0), axis);
    const ngraph::ResultVector results {std::make_shared<ngraph::opset1::Result>(softMax)};

    function = std::make_shared<ngraph::Function>(results, params, "softMax");
}
}  // namespace LayerTestsDefinitions