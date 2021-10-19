// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/conversion.hpp"

#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

std::string ConversionLayerTest::getTestCaseName(const testing::TestParamInfo<ConversionParamsTuple>& obj) {
    ngraph::helpers::ConversionTypes conversionOpType;
    InferenceEngine::Precision inputPrecision, targetPrecision;
    InferenceEngine::Layout inLayout, outLayout;
    std::string targetName;
    std::vector<std::vector<size_t>> inputShape;
    std::tie(conversionOpType, inputShape, inputPrecision, targetPrecision, inLayout, outLayout, targetName) =
        obj.param;
    std::ostringstream result;
    result << "conversionOpType=" << conversionNames[conversionOpType] << "_";
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "inputPRC=" << inputPrecision.name() << "_";
    result << "targetPRC=" << targetPrecision.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetName;
    return result.str();
}

void ConversionLayerTest::SetUp() {
    ngraph::helpers::ConversionTypes conversionOpType;
    InferenceEngine::Precision inputPrecision, targetPrecision;
    std::vector<std::vector<size_t>> inputShape;
    std::tie(conversionOpType, inputShape, inputPrecision, targetPrecision, inLayout, outLayout, targetDevice) =
        GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
    auto targetPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(targetPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, inputShape);
    std::shared_ptr<ngraph::Node> out;
    if (inputPrecision.is_float() && targetPrc.is_integral_number()) {
        // Add float number for float inputs
        auto add_constant = ov::op::v0::Constant::create(ngPrc, {1}, {0.9});
        auto add = std::make_shared<ov::op::v1::Add>(params.front(), add_constant);
        auto conversion = ngraph::builder::makeConversion(add, targetPrc, conversionOpType);
        // Add some operation after conversion to avoid unintentional conversion to float on legacy IE
        auto add_constant2 = ov::op::v0::Constant::create(targetPrc, {1}, {1});
        out = std::make_shared<ov::op::v1::Add>(conversion, add_constant2);
    } else {
        out = ngraph::builder::makeConversion(params.front(), targetPrc, conversionOpType);
    }

    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(out)};
    function = std::make_shared<ngraph::Function>(results, params, "Conversion");
}
}  // namespace LayerTestsDefinitions
