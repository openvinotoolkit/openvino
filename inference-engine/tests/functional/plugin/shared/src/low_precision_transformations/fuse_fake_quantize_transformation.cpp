// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fuse_fake_quantize_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>

namespace LayerTestsDefinitions {

std::string FuseFakeQuantizeTransformation::getTestCaseName(testing::TestParamInfo<FuseFakeQuantizeTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    std::tie(netPrecision, inputShapes, targetDevice, params, version, fakeQuantizeOnData) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params, version) << "_" << fakeQuantizeOnData;
    return result.str();
}

void FuseFakeQuantizeTransformation::SetUp() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    std::tie(netPrecision, inputShape, targetDevice, params, version, fakeQuantizeOnData) = this->GetParam();

    ConfigurePlugin(version);

    function = ngraph::builder::subgraph::FuseFakeQuantizeFunction::getOriginal(
        FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision),
        inputShape,
        toNGraph(params),
        fakeQuantizeOnData);

    ngraph::pass::InitNodeInfo().run_on_function(function);

    if (version == LptVersion::cnnNetwork) {
        validate();
    }
}

void FuseFakeQuantizeTransformation::validate() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    std::tie(netPrecision, inputShape, targetDevice, params, version, fakeQuantizeOnData) = this->GetParam();

    auto transformations = getLowPrecisionTransformations(params);
    transformations.removeCleanupTransformations("FakeQuantize");
    transformations.removeCleanupTransformations("ScaleShift");
    const InferenceEngine::CNNNetwork network = transform(transformations);

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = it->second->getCreatorLayer().lock();
    EXPECT_TRUE(outputLayer != nullptr);
    EXPECT_EQ("ScaleShift", outputLayer->type);

    EXPECT_EQ(1ul, outputLayer->insData.size());
    const InferenceEngine::DataPtr insData = outputLayer->insData[0].lock();
    EXPECT_TRUE(insData != nullptr);
    const InferenceEngine::CNNLayerPtr fakeQuantize = insData->getCreatorLayer().lock();
    EXPECT_TRUE(fakeQuantize != nullptr);
    EXPECT_EQ("FakeQuantize", fakeQuantize->type);

    if (params.updatePrecisions) {
        const InferenceEngine::Precision precision = fakeQuantize->outData[0]->getTensorDesc().getPrecision();
        EXPECT_TRUE((precision == InferenceEngine::Precision::U8) || (precision == InferenceEngine::Precision::I8));
    }

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(FuseFakeQuantizeTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
