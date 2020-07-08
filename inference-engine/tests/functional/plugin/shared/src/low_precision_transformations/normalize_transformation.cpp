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
#include "ngraph_functions/low_precision_transformations/normalize_l2_function.hpp"

namespace LayerTestsDefinitions {

std::string NormalizeTransformation::getTestCaseName(testing::TestParamInfo<NormalizeTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    std::pair<ngraph::Shape, ngraph::Shape> shapes;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    bool fuseMultiply;
    bool shift;
    std::tie(netPrecision, shapes, targetDevice, params, version, fuseMultiply, shift) = obj.param;

    std::ostringstream result;
    result << netPrecision.name() << "_" <<
        shapes.first << "_" <<
        shapes.second << "_" <<
        targetDevice << "_" <<
        toString(params) << "_" <<
        version <<
        (fuseMultiply ? "_multiply" : "") <<
        (shift ? "_shift" : "");
    return result.str();
}

void NormalizeTransformation::SetUp() {
    threshold = 3.e-3;
    std::pair<ngraph::Shape, ngraph::Shape> shapes;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    bool fuseMultiply;
    bool shift;
    std::tie(netPrecision, shapes, targetDevice, params, version, fuseMultiply, shift) = this->GetParam();

    ConfigurePlugin(version);

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    // const float low = params.precisionsOnActivations[0] == InferenceEngine::Precision::U8 ? (0.f + (shift ? 10.f : 0.f)) : (-128.f + (shift ? 10.f : 0.f));
    // const float high = params.precisionsOnActivations[0] == InferenceEngine::Precision::U8 ? 255.f : 127.f;
    // const float k = 10.f;

    // const auto paramNode = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape(inputShape));
    // const auto fakeQuantize = ngraph::builder::makeFakeQuantize(
    //    paramNode->output(0), ngPrc, 256, { 1ul },
    //    { low / k }, { high / k }, { low / k }, { high / k });

    // const auto axes = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{ 1 }, std::vector<int64_t>{ 1ul });
    // const auto normL2 = std::make_shared<ngraph::opset1::NormalizeL2>(fakeQuantize->output(0), axes, 1e-6, ngraph::op::EpsMode::ADD);

    // ngraph::ResultVector results;
    // if (fuseMultiply) {
    //    const auto multiplyConst = std::make_shared<ngraph::op::Constant>(
    //        ngPrc, ngraph::Shape{ inputShape[0], inputShape[1], 1ul, 1ul }, std::vector<float>{ 2.f });
    //    const auto multiply = std::make_shared<ngraph::opset1::Multiply>(normL2->output(0), multiplyConst);
    //    results = { std::make_shared<ngraph::opset1::Result>(multiply) };
    // } else {
    //    results = { std::make_shared<ngraph::opset1::Result>(normL2) };
    // }

    // function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector { paramNode }, "NormalizeTransformation");

    function = ngraph::builder::subgraph::NormalizeL2Function::getOriginal(
        ngPrc,
        shapes,
        FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(params.precisionsOnActivations[0]),
        fuseMultiply,
        shift);

    // TODO: move to some another place
    switch (version) {
        case LptVersion::cnnNetwork: {
            validateCNNNetwork();
            break;
        }
        case LptVersion::nGraph: {
            validateNGraph();
            break;
        }
        default: {
            THROW_IE_EXCEPTION << "unexpected LPT version " << version;
        }
    }
}

void NormalizeTransformation::validateCNNNetwork() {
    InferenceEngine::Precision netPrecision;
    std::pair<ngraph::Shape, ngraph::Shape> shapes;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    bool fuseMultiply;
    bool shift;
    std::tie(netPrecision, shapes, targetDevice, params, version, fuseMultiply, shift) = this->GetParam();

    const InferenceEngine::CNNNetwork network = transform(params);

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = getCreatorLayer(it->second).lock();
    EXPECT_TRUE(outputLayer != nullptr);
    EXPECT_EQ(shift ? "Normalize" : "ScaleShift", outputLayer->type);

    IE_SUPPRESS_DEPRECATED_END
}

void NormalizeTransformation::validateNGraph() {
    // TODO: remove: don't need to validate here

    // InferenceEngine::Precision netPrecision;
    // std::pair<ngraph::Shape, ngraph::Shape> shapes;
    // InferenceEngine::details::LayerTransformation::Params params;
    // LayerTestsUtils::LayerTransformation::LptVersion version;
    // bool fuseMultiply;
    // bool shift;
    // std::tie(netPrecision, shapes, targetDevice, params, version, fuseMultiply, shift) = this->GetParam();

    // std::vector<std::shared_ptr<ngraph::Function>> module{ function };
    // ngraph::pass::VisualizeTree("C:\\Projects\\temp\\test.original").run_on_module(module);

    // const std::shared_ptr<ngraph::Function> transformedFunction = transformNGraph(params);

    // std::vector<std::shared_ptr<ngraph::Function>> transformedModule{ transformedFunction };
    // ngraph::pass::VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(transformedModule);

    // auto res = compare_functions(f, f_ref);
    // ASSERT_TRUE(res.first) << res.second;
}

TEST_P(NormalizeTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
