// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"

#include "low_precision_transformations/concat_transformation.hpp"


namespace LayerTestsDefinitions {

std::string ConcatTransformation::getTestCaseName(testing::TestParamInfo<LayerTestsUtils::basicParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(netPrecision, inputShapes, targetDevice) = obj.param;

    std::ostringstream result;
    result << "inputShapes=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "netPrecision=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void ConcatTransformation::SetUp() {
    SetRefMode(LayerTestsUtils::RefMode::IE);

    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    std::tie(netPrecision, inputShape, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const auto paramNode1 = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape(inputShape));
    const auto fakeQuantize1 = makeFakeQuantize(paramNode1->output(0));

    const std::vector<size_t> inputShape2 = { inputShape[0], inputShape[1], inputShape[2] / 2, inputShape[3] / 2 };
    const auto paramNode2 = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape(inputShape2));

    const auto fakeQuantize2 = makeFakeQuantize(paramNode2->output(0));
    const auto interpolateShape = std::make_shared<ngraph::op::Constant>(
        ngraph::element::i64,
        ngraph::Shape{ 2 },
        std::vector<int64_t>({ static_cast<int64_t>(inputShape[2]), static_cast<int64_t>(inputShape[3]) }));
    ngraph::op::InterpolateAttrs interpolateAttrs;
    interpolateAttrs.align_corners = false;
    interpolateAttrs.antialias = false;
    interpolateAttrs.axes = ngraph::AxisSet{2, 3};
    interpolateAttrs.mode = "nearest";
    interpolateAttrs.pads_begin = { 0 };
    interpolateAttrs.pads_end = { 0 };
    const auto interpolate = std::make_shared<ngraph::opset1::Interpolate>(fakeQuantize2->output(0), interpolateShape, interpolateAttrs);

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize1->output(0), interpolate->output(0)}, 1);

    ngraph::ResultVector results {std::make_shared<ngraph::opset1::Result>(concat)};
    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector { paramNode1, paramNode2 }, "ConcatTransformation");

    // TODO: move to some another place
    validate();
}

std::shared_ptr<ngraph::opset1::FakeQuantize> ConcatTransformation::makeFakeQuantize(const ngraph::Output<ngraph::Node>& input) {
    auto inputLowConst = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape{ 1, 1, 1, 1 }, std::vector<float>({ 1.f }));
    auto inputHighConst = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape{ 1, 1, 1, 1 }, std::vector<float>({ 1.f }));
    auto outputLowConst = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape{ 1, 1, 1, 1 }, std::vector<float>({ 1.f }));
    auto outputHighConst = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape{ 1, 1, 1, 1 }, std::vector<float>({ 1.f }));
    auto fakeQuantize = std::make_shared<ngraph::opset1::FakeQuantize>(input, inputLowConst, inputHighConst, outputLowConst, outputHighConst, 256ul);
    return fakeQuantize;
}

void ConcatTransformation::validate() {
    const InferenceEngine::CNNNetwork network = transform();

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = it->second->getCreatorLayer().lock();
    EXPECT_TRUE(outputLayer != nullptr);
    EXPECT_EQ("ScaleShift", outputLayer->type);

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(ConcatTransformation, CompareWithRefImpl) {
    Run();

    if (targetDevice == std::string{CommonTestUtils::DEVICE_GPU}) {
        PluginCache::get().reset();
    }
};

}  // namespace LayerTestsDefinitions
