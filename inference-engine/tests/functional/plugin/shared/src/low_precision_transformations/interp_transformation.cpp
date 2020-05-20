// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/interp_transformation.hpp"

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

std::string InterpTransformation::getTestCaseName(testing::TestParamInfo<InterpTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params;
    bool align_corners;
    bool shift;
    std::tie(netPrecision, inputShapes, targetDevice, params, align_corners, shift) = obj.param;

    std::ostringstream result;
    result << netPrecision.name() << "_" <<
        CommonTestUtils::vec2str(inputShapes) << "_" <<
        targetDevice << "_" <<
        toString(params) << "_" <<
        (align_corners ? "_alignCorners" : "") <<
        (shift ? "_shift" : "");
    return result.str();
}

void InterpTransformation::SetUp() {
    SetRefMode(LayerTestsUtils::RefMode::IE);

    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    bool align_corners;
    bool shift;
    std::tie(netPrecision, inputShape, targetDevice, params, align_corners, shift) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const float low = params.precisionsOnActivations[0] == InferenceEngine::Precision::U8 ? (0.f + (shift ? 10.f : 0.f)) : (-128.f + (shift ? 10.f : 0.f));
    const float high = params.precisionsOnActivations[0] == InferenceEngine::Precision::U8 ? 255.f : 127.f;
    const float k = 10.f;

    const auto paramNode = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape(inputShape));
    const auto fakeQuantize = ngraph::builder::makeFakeQuantize(
        paramNode->output(0), ngPrc, 256, { 1ul },
        { low / k }, { high / k }, { low / k }, { high / k });

    const auto interpolateShape = std::make_shared<ngraph::op::Constant>(
        ngraph::element::i64,
        ngraph::Shape{ 2 },
        std::vector<int64_t>({ static_cast<int64_t>(2*inputShape[2]), static_cast<int64_t>(2*inputShape[3]) }));
    ngraph::op::InterpolateAttrs interpolateAttrs;
    interpolateAttrs.align_corners = align_corners;
    interpolateAttrs.antialias = false;
    interpolateAttrs.axes = ngraph::AxisSet{2, 3};
    interpolateAttrs.mode = "linear";
    interpolateAttrs.pads_begin = { 0 };
    interpolateAttrs.pads_end = { 0 };
    const auto interpolate = std::make_shared<ngraph::opset1::Interpolate>(fakeQuantize->output(0), interpolateShape, interpolateAttrs);

    ngraph::ResultVector results {std::make_shared<ngraph::opset1::Result>(interpolate)};
    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector { paramNode }, "InterpTransformation");

    // TODO: move to some another place
    validate();
}

void InterpTransformation::validate() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    bool align_corners;
    bool shift;
    std::tie(netPrecision, inputShape, targetDevice, params, align_corners, shift) = this->GetParam();

    const InferenceEngine::CNNNetwork network = transform(params);

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = it->second->getCreatorLayer().lock();
    EXPECT_TRUE(outputLayer != nullptr);
    if (params.precisionsOnActivations[0] == InferenceEngine::Precision::I8 &&
        params.precisionsOnActivations.size() == 1 && !shift) {
        EXPECT_EQ("Interp", outputLayer->type);
    } else {
        EXPECT_EQ("ScaleShift", outputLayer->type);
    }

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(InterpTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
