// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_with_neighbors_graph_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/low_precision_transformations/concat_function.hpp"

namespace LayerTestsDefinitions {

std::string ConcatWithNeighborsGraphTransformation::getTestCaseName(testing::TestParamInfo<ConcatNeighboringGraphTransformationParams> obj) {
    ngraph::element::Type_t precision;
    ngraph::Shape inputShapes;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::tie(precision, inputShapes, targetDevice, params, version) = obj.param;

    return getTestCaseNameByParams(precision, inputShapes, targetDevice, params, version);
}

InferenceEngine::Blob::Ptr ConcatWithNeighborsGraphTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    ngraph::element::Type_t netPrecision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::tie(netPrecision, inputShape, targetDevice, params, version) = this->GetParam();

    if ((info.name() != "input1") && (info.name() != "input2") && (info.name() != "input3")) {
        THROW_IE_EXCEPTION << "unexpected input name " << info.name();
    }
    const float k = (info.name() == "input1") ? 1.f : (info.name() == "input2" ? 2.f : 3.f);
    return LayerTransformation::GenerateInput(params.precisionsOnActivations[0], info.getTensorDesc(), k);
}

void ConcatWithNeighborsGraphTransformation::SetUp() {
    threshold = 2.e-2;
    ngraph::element::Type_t ngPrecision;
    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::tie(ngPrecision, inputShape, targetDevice, params, version) = this->GetParam();

    ConfigurePlugin(version);

    function = ngraph::builder::subgraph::ConcatFunction::getOriginalWithNeighbors(
        ngPrecision,
        inputShape,
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} },
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 2.f} },
        { 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f / 3.f} });

    if (version == LptVersion::cnnNetwork) {
        validate();
    }
}

void ConcatWithNeighborsGraphTransformation::validate() {
    ngraph::element::Type_t netPrecision;
    ngraph::Shape inputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::tie(netPrecision, inputShape, targetDevice, params, version) = this->GetParam();

    const InferenceEngine::CNNNetwork network = transform(toCNNNetwork(params));

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(2, outputs.size());

    for (const auto it : outputs) {
        const InferenceEngine::CNNLayerPtr outputLayer = getCreatorLayer(it.second).lock();
        EXPECT_TRUE(outputLayer != nullptr);
        EXPECT_EQ("ScaleShift", outputLayer->type);

        const InferenceEngine::CNNLayerPtr layer = InferenceEngine::details::CNNNetworkHelper::getParent(*outputLayer);
        if (params.updatePrecisions) {
            const auto interval = getQuantizationInterval(params.precisionsOnActivations[0]);
            const InferenceEngine::Precision expectedPrecision = interval.first >= 0.f ? InferenceEngine::Precision::U8 : InferenceEngine::Precision::I8;

            checkPrecisions(*layer, { { expectedPrecision }, { expectedPrecision } }, { { expectedPrecision } });
        } else {
            checkPrecisions(*layer, toCNNNetwork(netPrecision));
        }
    }

    // check quantized FQ layers map: should includes all FQ

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(ConcatWithNeighborsGraphTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
