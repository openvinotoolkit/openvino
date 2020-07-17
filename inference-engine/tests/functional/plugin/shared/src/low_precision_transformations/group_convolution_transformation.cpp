// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/group_convolution_transformation.hpp"

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
#include "ngraph_functions/low_precision_transformations/group_convolution_function.hpp"

namespace LayerTestsDefinitions {

std::string GroupConvolutionTransformation::getTestCaseName(testing::TestParamInfo<GroupConvolutionTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    GroupConvolutionTransformationParam param;
    std::tie(netPrecision, targetDevice, params, version, param) = obj.param;

    std::ostringstream result;
    result <<
        getTestCaseNameByParams(netPrecision, param.inputShape, targetDevice, params, version) << "_" <<
        param.inputShape << "_" <<
        param.outputShape << "_" <<
        param.group << "_" <<
        param.fakeQuantizeOnData << "_" <<
        param.fakeQuantizeOnWeights;
    return result.str();
}

void GroupConvolutionTransformation::SetUp() {
    threshold = 0.1f;

    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    GroupConvolutionTransformationParam param;
    std::tie(netPrecision, targetDevice, params, version, param) = this->GetParam();
    auto precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    ConfigurePlugin(version);

    function = ngraph::builder::subgraph::GroupConvolutionFunction::getOriginal(
        precision,
        param.inputShape,
        param.outputShape,
        param.group,
        param.fakeQuantizeOnData,
        param.fakeQuantizeOnWeights);

    if (version == LptVersion::cnnNetwork) {
        validate();
    }
}

void GroupConvolutionTransformation::validate() {
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    GroupConvolutionTransformationParam param;
    std::tie(netPrecision, targetDevice, params, version, param) = this->GetParam();

    const InferenceEngine::CNNNetwork network = transform(params);

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = getCreatorLayer(it->second).lock();
    EXPECT_TRUE(outputLayer != nullptr);
    EXPECT_EQ((!param.fakeQuantizeOnData.empty()) && (!param.fakeQuantizeOnWeights.empty()) ? "ScaleShift" : "Convolution", outputLayer->type);

    if ((!param.fakeQuantizeOnData.empty()) && (!param.fakeQuantizeOnWeights.empty())) {
        const InferenceEngine::CNNLayerPtr layer = InferenceEngine::details::CNNNetworkHelper::getParent(*outputLayer);
        if (params.updatePrecisions) {
            checkPrecisions(
                *layer,
                { { InferenceEngine::Precision::U8 }, { InferenceEngine::Precision::I8 } },
                { getDeviceInternalPrecision(netPrecision) });
        } else {
            checkPrecisions(*layer, netPrecision);
        }
    }

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(GroupConvolutionTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
