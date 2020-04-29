// Copyright (C) 2020 Intel Corporation
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
#include "ngraph/pass/visualize_tree.hpp"

#include "low_precision_transformations/normalize_transformation.hpp"

namespace LayerTestsDefinitions {

std::string NormalizeTransformation::getTestCaseName(testing::TestParamInfo<LayerTestsUtils::basicParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(netPrecision, inputShapes, targetDevice) = obj.param;

    std::ostringstream result;
    result << "inputShapes=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "netPrecision=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void NormalizeTransformation::SetUp() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    std::tie(netPrecision, inputShape, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const auto paramNode = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape(inputShape));
    const auto fakeQuantize = makeFakeQuantize(paramNode->output(0));

    const auto axes = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{ 3 }, std::vector<int64_t>{ 1, 2, 3 });
    const auto normL2 = std::make_shared<ngraph::opset1::NormalizeL2>(fakeQuantize->output(0), axes->output(0), 1e-6, ngraph::op::EpsMode::ADD);

    ngraph::ResultVector results {std::make_shared<ngraph::opset1::Result>(normL2)};
    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector { paramNode }, "NormalizeTransformation");

    // TODO: move to some another place
    validate();
}

std::shared_ptr<ngraph::opset1::FakeQuantize> NormalizeTransformation::makeFakeQuantize(const ngraph::Output<ngraph::Node>& input) {
    auto inputLowConst = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape{ 1, 1, 1, 1 }, std::vector<float>{ 0.f });
    auto inputHighConst = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape{ 1, 1, 1, 1 }, std::vector<float>{ 256.f });
    auto outputLowConst = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape{ 1, 1, 1, 1 }, std::vector<float>{ 0.f });
    auto outputHighConst = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape{ 1, 1, 1, 1 }, std::vector<float>{ 256.f / 2.f });
    auto fakeQuantize = std::make_shared<ngraph::opset1::FakeQuantize>(input, inputLowConst, inputHighConst, outputLowConst, outputHighConst, 256ul);
    return fakeQuantize;
}

IE_SUPPRESS_DEPRECATED_START

void NormalizeTransformation::validate() {
    const InferenceEngine::CNNNetwork network = transform();

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = it->second->getCreatorLayer().lock();
    EXPECT_TRUE(outputLayer != nullptr);
    EXPECT_EQ("ScaleShift", outputLayer->type);
}

IE_SUPPRESS_DEPRECATED_END

TEST_P(NormalizeTransformation, CompareWithRefImpl) {
    Run();

    if (targetDevice == std::string{CommonTestUtils::DEVICE_GPU}) {
        PluginCache::get().reset();
    }
};

}  // namespace LayerTestsDefinitions
