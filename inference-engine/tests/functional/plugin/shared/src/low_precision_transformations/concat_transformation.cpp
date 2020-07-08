// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "ngraph_functions/low_precision_transformations/concat_function.hpp"

namespace LayerTestsDefinitions {

std::string ConcatTransformation::getTestCaseName(testing::TestParamInfo<LayerTestsUtils::LayerTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::tie(netPrecision, inputShapes, targetDevice, params, version) = obj.param;

    return getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params, version);
}

InferenceEngine::Blob::Ptr ConcatTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::tie(netPrecision, inputShape, targetDevice, params, version) = this->GetParam();

    const float k = (info.name() == "input1") ? 1.f : (info.name() == "input2" ? 2.f : 3.f);
    return LayerTransformation::GenerateInput(params.precisionsOnActivations[0], info.getTensorDesc(), k);
}

void ConcatTransformation::SetUp() {
    // SetRefMode(LayerTestsUtils::RefMode::IE);

    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::tie(netPrecision, inputShape, targetDevice, params, version) = this->GetParam();

    ConfigurePlugin(version);

    function = ngraph::builder::subgraph::ConcatFunction::getOriginal(
        FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision),
        inputShape,
        toNGraph(params));

    if (version == LptVersion::cnnNetwork) {
        validate();
    }
}

void ConcatTransformation::validate() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::tie(netPrecision, inputShape, targetDevice, params, version) = this->GetParam();

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
