// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/normalize_transformation.hpp"

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
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

std::string NormalizeTransformation::getTestCaseName(testing::TestParamInfo<NormalizeTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params;
    bool fuseMultiply;
    bool shift;
    std::tie(netPrecision, inputShapes, targetDevice, params, fuseMultiply, shift) = obj.param;

    std::ostringstream result;
    result << netPrecision.name() << "_" <<
        CommonTestUtils::vec2str(inputShapes) << "_" <<
        targetDevice << "_" <<
        toString(params) << "_" <<
        (fuseMultiply ? "_multiply" : "") <<
        (shift ? "_shift" : "");
    return result.str();
}

void NormalizeTransformation::SetUp() {
    threshold = 3.e-3;
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    bool fuseMultiply;
    bool shift;
    std::tie(netPrecision, inputShape, targetDevice, params, fuseMultiply, shift) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const float low = params.precisionsOnActivations[0] == InferenceEngine::Precision::U8 ? (0.f + (shift ? 10.f : 0.f)) : (-128.f + (shift ? 10.f : 0.f));
    const float high = params.precisionsOnActivations[0] == InferenceEngine::Precision::U8 ? 255.f : 127.f;
    const float k = 10.f;

    const auto paramNode = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape(inputShape));
    const auto fakeQuantize = ngraph::builder::makeFakeQuantize(
        paramNode->output(0), ngPrc, 256, { 1ul },
        { low / k }, { high / k }, { low / k }, { high / k });

    const auto axes = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{ 1 }, std::vector<int64_t>{ 1ul });
    const auto normL2 = std::make_shared<ngraph::opset1::NormalizeL2>(fakeQuantize->output(0), axes, 1e-6, ngraph::op::EpsMode::ADD);

    ngraph::ResultVector results;
    if (fuseMultiply) {
        const auto multiplyConst = std::make_shared<ngraph::op::Constant>(ngPrc, ngraph::Shape(inputShape), std::vector<float>{ 2.f });
        const auto multiply = std::make_shared<ngraph::opset1::Multiply>(normL2->output(0), multiplyConst);
        results = { std::make_shared<ngraph::opset1::Result>(multiply) };
    } else {
        results = { std::make_shared<ngraph::opset1::Result>(normL2) };
    }

    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector { paramNode }, "NormalizeTransformation");

    // TODO: move to some another place
    validate();
}

void NormalizeTransformation::validate() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    bool fuseMultiply;
    bool shift;
    std::tie(netPrecision, inputShape, targetDevice, params, fuseMultiply, shift) = this->GetParam();

    const InferenceEngine::CNNNetwork network = transform(params);

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = it->second->getCreatorLayer().lock();
    EXPECT_TRUE(outputLayer != nullptr);
    EXPECT_EQ(shift ? "Normalize" : "ScaleShift", outputLayer->type);

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(NormalizeTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
