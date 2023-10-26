// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/reverse.hpp"

#include <ngraph/opsets/opset1.hpp>

#include "ov_models/builders.hpp"

using namespace InferenceEngine;
using namespace FuncTestUtils::PrecisionUtils;

namespace LayerTestsDefinitions {

std::string ReverseLayerTest::getTestCaseName(const testing::TestParamInfo<reverseParams>& obj) {
    std::vector<size_t> inputShape;
    std::vector<int> axes;
    std::string mode;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::tie(inputShape, axes, mode, netPrecision, targetDevice) = obj.param;

    std::ostringstream result;

    result << "in_shape=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "axes=" << ov::test::utils::vec2str(axes) << "_";
    result << "mode=" << mode << "_";
    result << "prec=" << netPrecision.name() << "_";
    result << "dev=" << targetDevice;
    return result.str();
}

void ReverseLayerTest::SetUp() {
    std::vector<size_t> inputShape;
    std::vector<int> axes;
    std::string mode;
    InferenceEngine::Precision netPrecision;
    std::tie(inputShape, axes, mode, netPrecision, targetDevice) = GetParam();

    const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ngraph::ParameterVector paramsVector;
    const ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    std::shared_ptr<ov::op::v0::Constant> axes_constant;
    if (mode == "index") {
        axes_constant = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{axes.size()}, axes);
    } else {
        std::vector<bool> axesMask(inputShape.size(), false);
        for (auto axe : axes) {
            axesMask[axe] = true;
        }
        axes_constant =
            std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{axesMask.size()}, axesMask);
    }
    const auto reverse = std::make_shared<ngraph::opset1::Reverse>(params[0], axes_constant, mode);
    function = std::make_shared<ngraph::Function>(reverse->outputs(), params, "reverse");
}
}  // namespace LayerTestsDefinitions
