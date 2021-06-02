// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/output_layers_handling_in_transformations_for_concat_multi_channel.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

std::pair<float, float> outputLayersHandlingInTransformationsForConcatMultiChannelGetInterval(const std::vector<ngraph::element::Type>& precisions) {
    const bool unsignedInterval = std::find(precisions.begin(), precisions.end(), ngraph::element::u8) != precisions.end();
    const float low = unsignedInterval ? 0.f : -128.f;
    const float hight = unsignedInterval ? 255.f : 127.f;
    return std::make_pair(low, hight);
}

std::string OutputLayersHandlingInTransformationsForConcatMultiChannel::getTestCaseName(
    testing::TestParamInfo<LayerTestsUtils::LayerTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, params) = obj.param;

    return getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params);
}

InferenceEngine::Blob::Ptr OutputLayersHandlingInTransformationsForConcatMultiChannel::GenerateInput(const InferenceEngine::InputInfo &info) const {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = this->GetParam();

    if ((info.name() != "input1") && (info.name() != "input2")) {
        IE_THROW() << "unexpected input name " << info.name();
    }
    const float k = (info.name() == "input1") ? 1.f : (info.name() == "input2" ? 2.f : 3.f);

    const auto interval = outputLayersHandlingInTransformationsForConcatMultiChannelGetInterval({ ngraph::element::u8, ngraph::element::i8 });
    const float low = interval.first / k;
    const float hight = interval.second / k;

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
*  Convolution/Power  Output
*        /
*       /
*   Output
*/

void OutputLayersHandlingInTransformationsForConcatMultiChannel::SetUp() {
    threshold = 0.05;

    InferenceEngine::SizeVector inputShape1;
    InferenceEngine::Precision netPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape1, targetDevice, params) = this->GetParam();

    auto ngPrecision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, ngraph::Shape(inputShape1));
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = ngraph::builder::makeFakeQuantize(input1->output(0), ngPrecision, 256ul, { 1ul });
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    ASSERT_EQ(4ul, inputShape1.size()) << "unexpected input layout";
    const InferenceEngine::SizeVector inputShape2 = { inputShape1[0], inputShape1[1] * 2ul, inputShape1[2], inputShape1[3] };
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = ngraph::builder::makeFakeQuantize(input2->output(0), ngPrecision, 256ul, { 1ul });
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0)}, 1);
    concat->set_friendly_name("concat");

    auto const1 = ngraph::opset1::Constant::create(ngPrecision, ngraph::Shape{ 1, 1, 1, 1 }, { 1 });
    std::shared_ptr<ngraph::opset1::Add> convolution = std::make_shared<ngraph::opset1::Add>(concat, const1);
    convolution->set_friendly_name("convolution");

    ngraph::ResultVector results {
        std::make_shared<ngraph::opset1::Result>(concat),
        std::make_shared<ngraph::opset1::Result>(convolution),
        std::make_shared<ngraph::opset1::Result>(fakeQuantize2)
    };

    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector { input1, input2 }, "OutputLayersHandling");
}

TEST_P(OutputLayersHandlingInTransformationsForConcatMultiChannel, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
