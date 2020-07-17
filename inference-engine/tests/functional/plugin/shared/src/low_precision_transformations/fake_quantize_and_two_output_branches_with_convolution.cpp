// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fake_quantize_and_two_output_branches_with_convolution.hpp"

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

// TODO: debug only
#include <ngraph/pass/visualize_tree.hpp>

namespace LayerTestsDefinitions {

std::string FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation::getTestCaseName(
    testing::TestParamInfo<FakeQuantizeAndTwoOutputBranchesWithConvolutionParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    ngraph::builder::subgraph::FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction::ActualValues testValues;
    std::tie(netPrecision, inputShapes, targetDevice, params, version, testValues) = obj.param;

    std::ostringstream result;
    result << netPrecision << "_" << targetDevice << "_" << version << "_" << testValues;
    return result.str();
}

void FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation::SetUp() {
    threshold = 0.1f;

    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    ngraph::builder::subgraph::FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction::ActualValues testValues;
    std::tie(netPrecision, inputShape, targetDevice, params, version, testValues) = this->GetParam();
    auto precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    ConfigurePlugin(version);

    function = ngraph::builder::subgraph::FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction::getOriginal(
        precision,
        inputShape,
        testValues);

    if (version == LptVersion::cnnNetwork) {
        validate();
    }
}

void FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation::validate() {
    //InferenceEngine::Precision netPrecision;
    //InferenceEngine::SizeVector inputShape;
    //std::string targetDevice;
    //InferenceEngine::details::LayerTransformation::Params params;
    //LayerTestsUtils::LayerTransformation::LptVersion version;
    //FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformationParams param;
    //std::tie(netPrecision, inputShape, targetDevice, params, version, param) = this->GetParam();

    //const InferenceEngine::CNNNetwork network = transform(params);

    //IE_SUPPRESS_DEPRECATED_START

    //InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    //EXPECT_EQ(1, outputs.size());

    //std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    //const InferenceEngine::CNNLayerPtr outputLayer = it->second->getCreatorLayer().lock();
    //EXPECT_TRUE(outputLayer != nullptr);
    //EXPECT_EQ((!param.fakeQuantizeOnData.empty()) && (!param.fakeQuantizeOnWeights.empty()) ? "ScaleShift" : "Convolution", outputLayer->type);

    //if ((!param.fakeQuantizeOnData.empty()) && (!param.fakeQuantizeOnWeights.empty())) {
    //    const InferenceEngine::CNNLayerPtr layer = InferenceEngine::details::CNNNetworkHelper::getParent(*outputLayer);
    //    if (params.updatePrecisions) {
    //        checkPrecisions(
    //            *layer,
    //            { { InferenceEngine::Precision::U8 }, { InferenceEngine::Precision::I8 } },
    //            { getDeviceInternalPrecision(netPrecision) });
    //    } else {
    //        checkPrecisions(*layer, netPrecision);
    //    }
    //}

    //IE_SUPPRESS_DEPRECATED_END
}

TEST_P(FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
