// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/output_layers_concat.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ov_models/pass/convert_prc.hpp"
#include "ov_models/builders.hpp"

namespace LayerTestsDefinitions {

std::string OutputLayersConcat::getTestCaseName(const testing::TestParamInfo<LayerTestsUtils::LayerTransformationParams>& obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, params) = obj.param;

    return getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params);
}

InferenceEngine::Blob::Ptr OutputLayersConcat::GenerateInput(const InferenceEngine::InputInfo &info) const {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = this->GetParam();

    if ((info.name() != "input1") && (info.name() != "input2")) {
        IE_THROW() << "unexpected input name " << info.name();
    }
    const float k = (info.name() == "input1") ? 1.f : 2.f;

    const float low = 0.f / k;
    const float hight = 255.f / k;
    InferenceEngine::Blob::Ptr input = FuncTestUtils::createAndFillBlobConsistently(info.getTensorDesc(), hight - low, static_cast<int32_t>(low), 1ul);
    return input;
}

/*
*           FQ1     FQ2
*            \      / \
*             \    /   Output
*             Concat
*            /      \
*           /        \
*  Convolution      Output
*        /
*       /
*   Output
*/

void OutputLayersConcat::SetUp() {
    InferenceEngine::SizeVector inputShape1;
    InferenceEngine::Precision netPrecision;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape1, targetDevice, params) = this->GetParam();

    auto ngPrecision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, ngraph::Shape(inputShape1));
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = ngraph::builder::makeFakeQuantize(
        input1->output(0), ngPrecision, 256ul, { 1ul },
        { 0.f }, { 255.f }, { 0.f }, { 255.f });
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    ASSERT_EQ(4ul, inputShape1.size()) << "unexpected input layout";
    const InferenceEngine::SizeVector inputShape2 = { inputShape1[0], inputShape1[1] * 2ul, inputShape1[2], inputShape1[3] };
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = ngraph::builder::makeFakeQuantize(
        input2->output(0), ngPrecision, 256ul, { 1ul },
        { 0.f }, { 255.f / 2.f }, { 0.f }, { 255.f / 2.f });
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0)}, 1);
    concat->set_friendly_name("concat");

    const float k = 1.f;
    const auto weights = ngraph::opset1::Constant::create(
        ngPrecision,
        ngraph::Shape{ inputShape1[1ul] + inputShape2[1ul], inputShape1[1ul] + inputShape2[1ul], 1ul, 1ul },
        std::vector<float>((inputShape1[1ul] + inputShape2[1ul]) * (inputShape1[1ul] + inputShape2[1ul]), 1ul));
    weights->set_friendly_name("weights");
    const auto fakeQuantizeOnWeights = ngraph::builder::makeFakeQuantize(
        weights, ngPrecision, 256ul, { 1ul },
        { -128.f / k }, { 127.f / k }, { -128.f / k }, { 127.f / k });
    fakeQuantizeOnWeights->set_friendly_name("fakeQuantizeOnWeights");

    const std::shared_ptr<ngraph::opset1::Convolution> convolution = std::make_shared<ngraph::opset1::Convolution>(
        concat->output(0),
        fakeQuantizeOnWeights,
        ngraph::Strides{ 1ul, 1ul },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1ul, 1ul });
    convolution->set_friendly_name("convolution");

    ngraph::ResultVector results {
        std::make_shared<ngraph::opset1::Result>(concat),
        std::make_shared<ngraph::opset1::Result>(convolution),
        std::make_shared<ngraph::opset1::Result>(fakeQuantize2)
    };

    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector { input1, input2 }, "OutputLayersHandling");
}

TEST_P(OutputLayersConcat, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
