// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "ov_models/builders.hpp"
#include "shared_test_classes/single_layer/gather_elements.hpp"

namespace LayerTestsDefinitions {

std::string GatherElementsLayerTest::getTestCaseName(const testing::TestParamInfo<GatherElementsParams>& obj) {
    InferenceEngine::SizeVector dataShape, indicesShape;
    InferenceEngine::Precision dPrecision, iPrecision;
    int axis;
    std::string device;
    std::tie(dataShape, indicesShape, axis, dPrecision, iPrecision, device) = obj.param;

    std::ostringstream result;
    result << "DS=" << ov::test::utils::vec2str(dataShape) << "_";
    result << "IS=" << ov::test::utils::vec2str(indicesShape) << "_";
    result << "Ax=" << axis << "_";
    result << "DP=" << dPrecision.name() << "_";
    result << "IP=" << iPrecision.name() << "_";
    result << "device=" << device;

    return result.str();
}

void GatherElementsLayerTest::SetUp() {
    InferenceEngine::SizeVector dataShape, indicesShape;
    InferenceEngine::Precision dPrecision, iPrecision;
    int axis;
    std::tie(dataShape, indicesShape, axis, dPrecision, iPrecision, targetDevice) = this->GetParam();
    outPrc = dPrecision;

    auto ngDPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(dPrecision);
    auto ngIPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(iPrecision);

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngDPrc, ov::Shape(dataShape))};
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto gather = std::dynamic_pointer_cast<ngraph::op::v6::GatherElements>(
            ngraph::builder::makeGatherElements(paramOuts[0], indicesShape, ngIPrc, axis));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(gather)};
    function = std::make_shared<ngraph::Function>(results, params, "gatherEl");
}

}  // namespace LayerTestsDefinitions
