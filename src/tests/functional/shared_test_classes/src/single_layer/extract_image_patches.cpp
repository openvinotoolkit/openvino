// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/extract_image_patches.hpp"
#include "ov_models/builders.hpp"


namespace LayerTestsDefinitions {

std::string ExtractImagePatchesTest::getTestCaseName(const testing::TestParamInfo<extractImagePatchesTuple> &obj) {
    std::vector<size_t> inputShape, kernel, strides, rates;
    ngraph::op::PadType pad_type;
    InferenceEngine::Precision netPrc;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout;
    std::string targetName;
    std::tie(inputShape, kernel, strides, rates, pad_type, netPrc, inPrc, outPrc, inLayout, targetName) = obj.param;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "netPRC=" << netPrc.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "K=" << ov::test::utils::vec2str(kernel) << "_";
    result << "S=" << ov::test::utils::vec2str(strides) << "_";
    result << "R=" << ov::test::utils::vec2str(rates) << "_";
    result << "P=" << pad_type << "_";
    result << "trgDev=" << targetName;
    return result.str();
}

void ExtractImagePatchesTest::SetUp() {
    std::vector<size_t> inputShape, kernel, strides, rates;
    ngraph::op::PadType pad_type;
    InferenceEngine::Precision netPrecision;
    std::tie(inputShape, kernel, strides, rates, pad_type, netPrecision, inPrc, outPrc, inLayout, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto inputNode = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape(inputShape));
    ngraph::ParameterVector params = {inputNode};

    auto extImgPatches = std::make_shared<ngraph::opset3::ExtractImagePatches>(
        inputNode, ngraph::Shape(kernel), ngraph::Strides(strides), ngraph::Shape(rates), pad_type);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(extImgPatches)};
    function = std::make_shared<ngraph::Function>(results, params, "ExtractImagePatches");
}
}  // namespace LayerTestsDefinitions
