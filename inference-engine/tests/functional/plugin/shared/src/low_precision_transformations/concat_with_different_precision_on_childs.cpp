// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_with_different_precision_on_childs.hpp"

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

std::string ConcatWithDifferentChildsTransformation::getTestCaseName(testing::TestParamInfo<ConcatWithDifferentChildsTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShapes;
    std::string targetDevice;
    ConcatWithDifferentChildsTransformationParam param;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    bool multiChannel;
    std::tie(netPrecision, inputShapes, targetDevice, param, params, version, multiChannel) = obj.param;

    std::ostringstream result;
    result <<
        getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params, version) <<
        (multiChannel ? "_multichannel" : "") << param.fqOnData1 << param.fqOnData2;

    return result.str();
}

InferenceEngine::Blob::Ptr ConcatWithDifferentChildsTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShapes;
    std::string targetDevice;
    ConcatWithDifferentChildsTransformationParam param;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    bool multiChannel;
    std::tie(netPrecision, inputShapes, targetDevice, param, params, version, multiChannel) = this->GetParam();

    const float k = (info.name() == "input1") ? 1.f : (info.name() == "input2" ? 2.f : 3.f);
    return LayerTransformation::GenerateInput(params.precisionsOnActivations[0], info.getTensorDesc(), k);
}

void ConcatWithDifferentChildsTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShapes;
    ConcatWithDifferentChildsTransformationParam param;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    bool multiChannel;
    std::tie(netPrecision, inputShapes, targetDevice, param, params, version, multiChannel) = this->GetParam();

    ConfigurePlugin(version);

    function = ngraph::builder::subgraph::ConcatFunction::getOriginalWithDifferentPrecisionOnChilds(
        netPrecision, inputShapes, param.fqOnData1, param.fqOnData2);
    if (version == LptVersion::cnnNetwork) {
        validate();
    }
}

void ConcatWithDifferentChildsTransformation::validate() {
    ngraph::element::Type netPrecision;
    ngraph::Shape inputShape;
    ConcatWithDifferentChildsTransformationParam param;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    bool multichannel;
    std::tie(netPrecision, inputShape, targetDevice, param, params, version, multichannel) = this->GetParam();

    InferenceEngine::details::LowPrecisionTransformations transformations = getLowPrecisionTransformations(toCNNNetwork(params));
    if (!multichannel) {
        transformations.addBranchSpecific<InferenceEngine::details::ConcatTransformation>(toCNNNetwork(params), "Concat");
    }
    const InferenceEngine::CNNNetwork network = transform(transformations);

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(2, outputs.size());

    const InferenceEngine::CNNLayerPtr outputLayer0 = getCreatorLayer(outputs.begin()->second).lock();
    const InferenceEngine::CNNLayerPtr outputLayer1 = getCreatorLayer((++outputs.begin())->second).lock();
    EXPECT_EQ("ScaleShift", outputLayer0->type);
    EXPECT_EQ("ScaleShift", outputLayer1->type);

    const InferenceEngine::CNNLayerPtr layer = InferenceEngine::details::CNNNetworkHelper::getParent(*outputLayer1);
    if (params.updatePrecisions) {
        const auto interval = getQuantizationInterval(params.precisionsOnActivations[0]);
        const InferenceEngine::Precision expectedPrecision = interval.first >= 0.f ? InferenceEngine::Precision::U8 : InferenceEngine::Precision::I8;

        checkPrecisions(*layer, { { expectedPrecision } }, { { expectedPrecision }, { expectedPrecision } });
    } else {
        checkPrecisions(*layer, toCNNNetwork(netPrecision));
    }
    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(ConcatWithDifferentChildsTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
