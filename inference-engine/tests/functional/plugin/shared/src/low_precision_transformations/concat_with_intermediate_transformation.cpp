// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_with_intermediate_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include "low_precision_transformations/concat.hpp"

#include <transformations/init_node_info.hpp>
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/low_precision_transformations/concat_function.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace LayerTestsDefinitions {

std::string ConcatWithIntermediateTransformation::getTestCaseName(testing::TestParamInfo<ConcatWithIntermediateTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    bool transparentIntermediate;
    bool multichannel;
    std::tie(netPrecision, inputShapes, targetDevice, params, version, transparentIntermediate, multichannel) = obj.param;

    std::ostringstream result;
    result <<
        getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params, version) <<
        (transparentIntermediate ? "" : "_notTransparentIntermediate") <<
        (multichannel ? "_multichannel" : "");

    return result.str();
}

InferenceEngine::Blob::Ptr ConcatWithIntermediateTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    ngraph::element::Type netPrecision;
    InferenceEngine::SizeVector inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params trasformationParams;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    bool transparentIntermediate;
    bool multichannel;
    std::tie(netPrecision, inputShape, targetDevice, trasformationParams, version, transparentIntermediate, multichannel) = this->GetParam();

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
    ngraph::element::Type ngPrecision;
    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params trasformationParams;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    bool transparentIntermediate;
    bool multichannel;
    std::tie(ngPrecision, inputShape, targetDevice, trasformationParams, version, transparentIntermediate, multichannel) = this->GetParam();

    ConfigurePlugin(version);

    function = ngraph::builder::subgraph::ConcatFunction::getOriginalWithIntermediate(
        ngPrecision,
        inputShape,
        transparentIntermediate,
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 2.f} });

    if (version == LptVersion::cnnNetwork) {
        validate();
    }
}

void ConcatWithIntermediateTransformation::validate() {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    bool transparentIntermediate;
    bool multichannel;
    std::tie(netPrecision, inputShape, targetDevice, params, version, transparentIntermediate, multichannel) = this->GetParam();

    InferenceEngine::details::LowPrecisionTransformations transformations = getLowPrecisionTransformations(toCNNNetwork(params));
    if (!multichannel) {
        transformations.addBranchSpecific<InferenceEngine::details::ConcatTransformation>(toCNNNetwork(params), "Concat");
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
            checkPrecisions(*layer, toCNNNetwork(netPrecision));
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
