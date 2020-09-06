// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_transformation.hpp"

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
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

std::string ConcatTransformation::getTestCaseName(testing::TestParamInfo<LayerTestsUtils::LayerTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, params) = obj.param;

    std::ostringstream result;
    result << netPrecision.name() << "_" << targetDevice << "_" << toString(params);
    return result.str();
}

InferenceEngine::Blob::Ptr ConcatTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = this->GetParam();

    const float k = (info.name() == "input1") ? 1.f : (info.name() == "input2" ? 2.f : 3.f);
    return LayerTransformation::GenerateInput(params.precisionsOnActivations[0], info.getTensorDesc(), k);
}

void ConcatTransformation::SetUp() {
    SetRefMode(LayerTestsUtils::RefMode::IE);

    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = this->GetParam();
    const auto ngPrecision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const auto interval = getQuantizationInterval(params.precisionsOnActivations[0]);
    const float low = interval.first;
    const float hight = interval.second;

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, ngraph::Shape(inputShape));
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = ngraph::builder::makeFakeQuantize(
        input1, ngPrecision, 256ul, { 1ul },
        { low }, { hight }, { low }, { hight });

    const std::vector<size_t> inputShape2 = { inputShape[0], inputShape[1], inputShape[2] / 2, inputShape[3] / 2 };
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, ngraph::Shape(inputShape2));
    input1->set_friendly_name("input2");
    const auto fakeQuantize2 = ngraph::builder::makeFakeQuantize(
        input2, ngPrecision, 256ul, { 1ul },
        { low / 2.f }, { hight / 2.f }, { low / 2.f }, { hight / 2.f });

    const auto interpolateShape = std::make_shared<ngraph::op::Constant>(
        ngraph::element::i64,
        ngraph::Shape{ 2 },
        std::vector<int64_t>({ static_cast<int64_t>(inputShape[2]), static_cast<int64_t>(inputShape[3]) }));
    ngraph::op::v0::InterpolateAttrs interpolateAttrs;
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
    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector { input1, input2 }, "ConcatTransformation");

    // TODO: move to some another place
    validate();
}

void ConcatTransformation::validate() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = this->GetParam();

    const InferenceEngine::CNNNetwork network = transform(params);

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = getCreatorLayer(it->second).lock();
    EXPECT_TRUE(outputLayer != nullptr);
    EXPECT_EQ("ScaleShift", outputLayer->type);

    const InferenceEngine::CNNLayerPtr layer = InferenceEngine::details::CNNNetworkHelper::getParent(*outputLayer);
    if (params.updatePrecisions) {
        const auto interval = getQuantizationInterval(params.precisionsOnActivations[0]);
        const InferenceEngine::Precision expectedPrecision = interval.first >= 0.f ? InferenceEngine::Precision::U8 : InferenceEngine::Precision::I8;
        checkPrecisions(*layer, { { expectedPrecision }, { expectedPrecision } }, { { expectedPrecision } });
    } else {
        checkPrecisions(*layer, netPrecision);
    }

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(ConcatTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
