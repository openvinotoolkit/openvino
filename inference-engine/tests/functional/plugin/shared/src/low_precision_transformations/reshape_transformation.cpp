// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/reshape_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include "ngraph_functions/builders.hpp"
#include <transformations/init_node_info.hpp>
#include "ngraph_functions/low_precision_transformations/reshape_function.hpp"


namespace LayerTestsDefinitions {

std::string ReshapeTransformation::getTestCaseName(testing::TestParamInfo<ReshapeTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    std::string targetDevice;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    ReshapeTransformationParam param;
    std::tie(netPrecision, targetDevice, params, version, param) = obj.param;

    std::ostringstream result;
    result << netPrecision << "_" << targetDevice << "_" << version << "_" << toString(params) <<
        "_" << param.inputShape << "_" << param.fakeQuantize << "_{";
    for (size_t i = 0; i < param.reshapeConstValues.size(); ++i) {
        result << param.reshapeConstValues[i];
        if (i != (param.reshapeConstValues.size() - 1ul)) {
            result << ", ";
        }
    }
    result << " }";
    return result.str();
}

void ReshapeTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    ReshapeTransformationParam param;
    std::tie(netPrecision, targetDevice, params, version, param) = this->GetParam();

    ConfigurePlugin(version);
    function = ngraph::builder::subgraph::ReshapeFunction::getOriginal(
        param.inputShape,
        param.reshapeConstValues,
        netPrecision,
        param.fakeQuantize);

    if (version == LptVersion::cnnNetwork) {
        validate();
    }
}

void ReshapeTransformation::validate() {
    ngraph::element::Type netPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params tmp;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    ReshapeTransformationParam param;
    std::tie(netPrecision, targetDevice, tmp, version, param) = this->GetParam();
    InferenceEngine::Precision precision = toCNNNetwork(netPrecision);

    const auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    const InferenceEngine::CNNNetwork network = transform(toCNNNetwork(params));

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = getCreatorLayer(it->second).lock();
    EXPECT_TRUE(outputLayer != nullptr);
    EXPECT_EQ("ScaleShift", outputLayer->type);
    auto expectedPrecision = getDeviceInternalPrecision(precision);
    const InferenceEngine::CNNLayerPtr layer = InferenceEngine::details::CNNNetworkHelper::getParent(*outputLayer);
    if (params.updatePrecisions) {
        const auto interval = getQuantizationInterval(params.precisionsOnActivations[0]);
        const InferenceEngine::Precision expectedPrecision = interval.first >= 0.f ? InferenceEngine::Precision::U8 : InferenceEngine::Precision::I8;
        checkPrecisions(*layer, { expectedPrecision });
    } else {
        checkPrecisions(*layer, precision);
    }

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(ReshapeTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
