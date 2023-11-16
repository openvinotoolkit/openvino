// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "ov_models/builders.hpp"
#include "shared_test_classes/single_layer/gather_elements.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

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

    int posAxis = axis;
    if (posAxis < 0)
        posAxis += dataShape.size();
    const auto axisDim = dataShape[posAxis];

    auto indicesValues = ov::test::utils::create_and_fill_tensor(ov::element::i32, indicesShape, axisDim - 1, 0);
    auto indicesNode = std::make_shared<ov::op::v0::Constant>(indicesValues);

    auto gather = std::make_shared<ov::op::v6::GatherElements>(params[0], indicesNode, axis);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(gather)};
    function = std::make_shared<ngraph::Function>(results, params, "gatherEl");
}

}  // namespace LayerTestsDefinitions
