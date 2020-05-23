// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

#include "low_precision_transformations/concat_neighboring_graph_transformation.hpp"


namespace LayerTestsDefinitions {

std::string ConcatNeighboringGraphTransformation::getTestCaseName(testing::TestParamInfo<LayerTestsUtils::LayerTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, params) = obj.param;

    std::ostringstream result;
    result << netPrecision.name() << "_" << targetDevice << "_" << toString(params);
    return result.str();
}

void ConcatNeighboringGraphTransformation::SetUp() {
    SetRefMode(LayerTestsUtils::RefMode::IE);

    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = this->GetParam();
    auto ngPrecision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const auto paramNode1 = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, ngraph::Shape(inputShape));
    const auto fakeQuantize1 = makeFakeQuantize(ngPrecision, paramNode1->output(0));

    const auto paramNode2 = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, ngraph::Shape(inputShape));
    const auto fakeQuantize2 = makeFakeQuantize(ngPrecision, paramNode2->output(0));

    const auto paramNode3 = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, ngraph::Shape(inputShape));
    const auto fakeQuantize3 = makeFakeQuantize(ngPrecision, paramNode3->output(0));

    const std::shared_ptr<ngraph::opset1::Concat> concat1 = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{
        fakeQuantize1->output(0),
        fakeQuantize2->output(0) },
        1ull);

    const std::shared_ptr<ngraph::opset1::Concat> concat2 = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{
        fakeQuantize2->output(0),
        fakeQuantize3->output(0) },
        1ull);

    const ngraph::ResultVector results {
        std::make_shared<ngraph::opset1::Result>(concat1),
        std::make_shared<ngraph::opset1::Result>(concat2)
    };

    function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector { paramNode1, paramNode2, paramNode3 },
        "ConcatNeighboringGraphTransformation");

    // TODO: move to some another place
    validate();
}

void ConcatNeighboringGraphTransformation::validate() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = this->GetParam();

    const InferenceEngine::CNNNetwork network = transform(params);

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(2, outputs.size());

    for (const auto it : outputs) {
        const InferenceEngine::CNNLayerPtr outputLayer = it.second->getCreatorLayer().lock();
        EXPECT_TRUE(outputLayer != nullptr);
        EXPECT_EQ("ScaleShift", outputLayer->type);
    }

    // check quantized FQ layers map: should includes all FQ

    IE_SUPPRESS_DEPRECATED_START
}

TEST_P(ConcatNeighboringGraphTransformation, CompareWithRefImpl) {
    Run();

    if (targetDevice == std::string{CommonTestUtils::DEVICE_GPU}) {
        PluginCache::get().reset();
    }
};

}  // namespace LayerTestsDefinitions
