// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/gemm_transformation.hpp"

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

std::string GemmTransformation::getTestCaseName(testing::TestParamInfo<LayerTestsUtils::LayerTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, params) = obj.param;

    std::ostringstream result;
    result << inputShapes.size() << "D_" << netPrecision.name() << "_" << targetDevice << "_" << toString(params);
    return result.str();
}

void GemmTransformation::SetUp() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = this->GetParam();
    auto ngPrecision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const auto paramNode1 = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, ngraph::Shape(inputShape));
    const auto fakeQuantizeOnAcitvations = ngraph::builder::makeFakeQuantize(
        paramNode1, ngPrecision, 256ul, { 1ul },
        { 0.f }, { 255.f / 4.f }, { 0.f }, { 255.f / 4.f });
    fakeQuantizeOnAcitvations->set_friendly_name("fakeQuantizeOnAcitvations");

    const auto paramNode2 = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, ngraph::Shape(inputShape));
    const auto fakeQuantizeOnWeights = ngraph::builder::makeFakeQuantize(
        paramNode2, ngPrecision, 256ul, { 1ul },
        { -128.f / 8.f }, { 127.f / 8.f }, { -128.f / 8.f }, { 127.f / 8.f });
    fakeQuantizeOnWeights->set_friendly_name("fakeQuantizeOnWeights");

    const auto matMul = std::make_shared<ngraph::opset1::MatMul>(
        fakeQuantizeOnAcitvations->output(0),
        fakeQuantizeOnWeights->output(0),
        false,
        false);
    matMul->set_friendly_name("matMul");

    ngraph::ResultVector results {std::make_shared<ngraph::opset1::Result>(matMul)};
    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector { paramNode1, paramNode2 }, "GemmTransformation");

    // TODO: move to some another place
    validate();
}

void GemmTransformation::validate() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = this->GetParam();

    const InferenceEngine::CNNNetwork network = transform(params);

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    for (const auto it : outputs) {
        const InferenceEngine::CNNLayerPtr outputLayer = it.second->getCreatorLayer().lock();
        EXPECT_TRUE(outputLayer != nullptr);
        EXPECT_EQ("ScaleShift", outputLayer->type);

        checkParentPrecision(outputLayer, false);
    }

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(GemmTransformation, CompareWithRefImpl) {
    Run();

    if (targetDevice == std::string{CommonTestUtils::DEVICE_GPU}) {
        PluginCache::get().reset();
    }
};

}  // namespace LayerTestsDefinitions
