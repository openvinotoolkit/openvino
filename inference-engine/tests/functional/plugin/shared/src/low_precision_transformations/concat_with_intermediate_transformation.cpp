// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_with_intermediate_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>
#include <ie_common.h>

#include "ngraph_functions/pass/convert_prc.hpp"
#include "low_precision_transformations/network_helper.hpp"
#include "low_precision_transformations/concat.hpp"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/builders.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace LayerTestsDefinitions {

std::string ConcatWithIntermediateTransformation::getTestCaseName(testing::TestParamInfo<ConcatWithIntermediateTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params;
    bool transparentIntermediate;
    bool multichannel;
    std::tie(netPrecision, inputShapes, targetDevice, params, transparentIntermediate, multichannel) = obj.param;

    std::ostringstream result;
    result <<
        netPrecision.name() << "_" <<
        targetDevice << "_" <<
        toString(params) <<
        (transparentIntermediate ? "" : "_notTransparentIntermediate") <<
        (multichannel ? "_multichannel" : "");

    return result.str();
}

InferenceEngine::Blob::Ptr ConcatWithIntermediateTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    InferenceEngine::SizeVector inputShape;
    std::string targetDevice;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params trasformationParams;
    bool transparentIntermediate;
    bool multichannel;
    std::tie(netPrecision, inputShape, targetDevice, trasformationParams, transparentIntermediate, multichannel) = this->GetParam();

    const float k = (info.name() == "input1") ? 1.f : (info.name() == "input2" ? 2.f : 3.f);
    return LayerTransformation::GenerateInput(trasformationParams.precisionsOnActivations[0], info.getTensorDesc(), k);
}

/*
* FQ       FQ
*  \       /
*   \  Intermediate (MaxPooling or Convolution)
*    \  /    \
*   Concat   Convolution
*/

void ConcatWithIntermediateTransformation::SetUp() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params trasformationParams;
    bool transparentIntermediate;
    bool multichannel;
    std::tie(netPrecision, inputShape, targetDevice, trasformationParams, transparentIntermediate, multichannel) = this->GetParam();
    const auto ngPrecision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const std::vector<size_t> inputShape1 = {
        inputShape[0],
        inputShape[1],
        inputShape[2] - (transparentIntermediate ? 2 : 0),
        inputShape[3] - (transparentIntermediate ? 2 : 0)
    };

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, ngraph::Shape(inputShape1));
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = ngraph::builder::makeFakeQuantize(input1, ngPrecision, 256ul, { 1ul }, { 0.f }, { 3.f }, { 0.f }, { 3.f });
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const std::vector<size_t> inputShape2 = { inputShape[0], inputShape[1], inputShape[2], inputShape[3] };
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = ngraph::builder::makeFakeQuantize(input2, ngPrecision, 256ul, { 1ul }, { 0.f }, { 9.f }, { 0.f }, { 9.f });
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    const std::vector<size_t> kernel = { 3, 3 };
    const std::vector<size_t> stride = { 1, 1 };
    const std::vector<size_t> padBegin = { 0, 0 };
    const std::vector<size_t> padEnd = { 0, 0 };
    const ngraph::op::PadType padType = ngraph::op::PadType::NOTSET;
    const ngraph::op::RoundingType roundingType = ngraph::op::RoundingType::FLOOR;
    std::shared_ptr<ngraph::op::Op> intermediateOp;

    if (transparentIntermediate) {
        intermediateOp = std::make_shared<ngraph::opset1::MaxPool>(
            fakeQuantize2->output(0),
            stride,
            padBegin,
            padEnd,
            kernel,
            roundingType,
            padType);
    } else {
        auto weights = ngraph::opset1::Constant::create(
            ngPrecision,
            ngraph::Shape{ inputShape[1], inputShape[1], 1, 1 },
            std::vector<float>(inputShape[1] * inputShape[1], 1));

        intermediateOp = std::make_shared<ngraph::opset1::Convolution>(
            fakeQuantize2->output(0),
            weights,
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });
    }

    intermediateOp->set_friendly_name("intermediate");

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize1->output(0), intermediateOp->output(0) }, 1);
    concat->set_friendly_name("concat");


    auto weights = ngraph::opset1::Constant::create(ngPrecision, ngraph::Shape{ inputShape[1], inputShape[1], 1, 1 }, { 1 });
    auto convolution = std::make_shared<ngraph::opset1::Convolution>(
        intermediateOp,
        weights,
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });
    convolution->set_friendly_name("convolution");

    ngraph::ResultVector results {
        std::make_shared<ngraph::opset1::Result>(concat),
        std::make_shared<ngraph::opset1::Result>(convolution)
    };
    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector { input1, input2 }, "ConcatWithIntermediateTransformation");

    // TODO: move to some another place
    validate();
}

void ConcatWithIntermediateTransformation::validate() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    bool transparentIntermediate;
    bool multichannel;
    std::tie(netPrecision, inputShape, targetDevice, params, transparentIntermediate, multichannel) = this->GetParam();

    InferenceEngine::details::LowPrecisionTransformations transformations = getLowPrecisionTransformations(params);
    if (!multichannel) {
        transformations.addBranchSpecific<InferenceEngine::details::ConcatTransformation>(params, "Concat");
    }
    const InferenceEngine::CNNNetwork network = transform(transformations);

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(2, outputs.size());

    const CNNLayerPtr intermediate = CNNNetworkHelper::getLayer(network, "intermediate");
    if (transparentIntermediate) {
        std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*intermediate);
        EXPECT_EQ(2ul, children.size());
        EXPECT_TRUE(
            ((children[0]->type == "ScaleShift") && (children[1]->type == "Concat")) ||
            ((children[0]->type == "Concat") && (children[1]->type == "ScaleShift")));

        const CNNLayerPtr concat = CNNNetworkHelper::getLayer(network, "concat_original");
        children = CNNNetworkHelper::getChildren(*concat);
        EXPECT_EQ(1ul, children.size());
        EXPECT_EQ("ScaleShift", children[0]->type);

        const InferenceEngine::CNNLayerPtr layer = InferenceEngine::details::CNNNetworkHelper::getParent(*children[0]);
        if (params.updatePrecisions) {
            const auto interval = getQuantizationInterval(params.precisionsOnActivations[0]);
            const InferenceEngine::Precision expectedPrecision = interval.first >= 0.f ? InferenceEngine::Precision::U8 : InferenceEngine::Precision::I8;
            checkPrecisions(*layer, { { expectedPrecision }, { expectedPrecision } }, { { expectedPrecision } });
        } else {
            checkPrecisions(*layer, netPrecision);
        }
    } else {
        std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*intermediate);
        EXPECT_EQ(2ul, children.size());
        EXPECT_TRUE(
            ((children[0]->type == "Convolution") && (children[1]->type == "Concat")) ||
            ((children[0]->type == "Concat") && (children[1]->type == "Convolution")));

        const CNNLayerPtr concat = CNNNetworkHelper::getLayer(network, "concat");
        children = CNNNetworkHelper::getChildren(*concat);
        EXPECT_EQ(0ul, children.size());
    }

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(ConcatWithIntermediateTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
