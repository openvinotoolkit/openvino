// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_neighboring_graph_transformation.hpp"

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

InferenceEngine::Blob::Ptr ConcatNeighboringGraphTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = this->GetParam();

    if ((info.name() != "input1") && (info.name() != "input2") && (info.name() != "input3")) {
        THROW_IE_EXCEPTION << "unexpected input name " << info.name();
    }
    const float k = (info.name() == "input1") ? 1.f : (info.name() == "input2" ? 2.f : 3.f);
    return LayerTransformation::GenerateInput(params.precisionsOnActivations[0], info.getTensorDesc(), k);
}

void ConcatNeighboringGraphTransformation::SetUp() {
    threshold = 2.e-2;
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = this->GetParam();
    const auto ngPrecision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const auto interval = getQuantizationInterval(params.precisionsOnActivations[0]);
    const float low = interval.first;
    const float hight = interval.second;

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, ngraph::Shape(inputShape));
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = ngraph::builder::makeFakeQuantize(
        input1, ngPrecision, 256ul, { 1ul },
        { low }, { hight }, { low }, { hight });
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, ngraph::Shape(inputShape));
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = ngraph::builder::makeFakeQuantize(
        input2, ngPrecision, 256ul, { 1ul },
        { low / 2.f }, { hight / 2.f }, { low / 2.f }, { hight / 2.f });
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    const auto input3 = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, ngraph::Shape(inputShape));
    input3->set_friendly_name("input3");
    const auto fakeQuantize3 = ngraph::builder::makeFakeQuantize(
        input3, ngPrecision, 256ul, { 1ul },
        { low / 3.f }, { hight / 3.f }, { low / 3.f }, { hight / 3.f });
    fakeQuantize3->set_friendly_name("fakeQuantize3");

    const auto concat1 = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{
        fakeQuantize1->output(0),
        fakeQuantize2->output(0) },
        1ull);
    concat1->set_friendly_name("concat1");

    const auto concat2 = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{
        fakeQuantize2->output(0),
        fakeQuantize3->output(0) },
        1ull);
    concat2->set_friendly_name("concat2");

    const ngraph::ResultVector results {
        std::make_shared<ngraph::opset1::Result>(concat1),
        std::make_shared<ngraph::opset1::Result>(concat2)
    };

    function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector { input1, input2, input3 },
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

        const InferenceEngine::CNNLayerPtr layer = InferenceEngine::details::CNNNetworkHelper::getParent(*outputLayer);
        if (params.updatePrecisions) {
            const auto interval = getQuantizationInterval(params.precisionsOnActivations[0]);
            const InferenceEngine::Precision expectedPrecision = interval.first >= 0.f ? InferenceEngine::Precision::U8 : InferenceEngine::Precision::I8;

            checkPrecisions(*layer, { { expectedPrecision }, { expectedPrecision } }, { { expectedPrecision } });
        } else {
            checkPrecisions(*layer, netPrecision);
        }
    }

    // check quantized FQ layers map: should includes all FQ

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(ConcatNeighboringGraphTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
