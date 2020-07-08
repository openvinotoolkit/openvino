// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/depth_to_space_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include <transformations/depth_to_space_fusion.hpp>
#include "ngraph_functions/low_precision_transformations/depth_to_space_function.hpp"

namespace LayerTestsDefinitions {

std::string DepthToSpaceTransformation::getTestCaseName(testing::TestParamInfo<DepthToSpaceTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::tie(netPrecision, inputShapes, targetDevice, params, version) = obj.param;

    std::ostringstream result;
    result << netPrecision.name() << "_" << targetDevice << "_" << version << "_" << toString(params);
    return result.str();
}

void DepthToSpaceTransformation::SetUp() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::tie(netPrecision, inputShape, targetDevice, params, version) = this->GetParam();

    if (inputShape.size() != 4ul) {
        THROW_IE_EXCEPTION << "not supported input shape size " << inputShape.size();
    }

    ConfigurePlugin(version);

    const auto ngPrecision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    function = ngraph::builder::subgraph::DepthToSpaceFunction::getOriginal(ngPrecision, inputShape);

    ngraph::pass::InitNodeInfo().run_on_function(function);
    ngraph::pass::DepthToSpaceFusion().run_on_function(function);

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

void DepthToSpaceTransformation::validateCNNNetwork() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::tie(netPrecision, inputShape, targetDevice, params, version) = this->GetParam();

    const InferenceEngine::CNNNetwork network = transform(params);

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = getCreatorLayer(it->second).lock();
    EXPECT_TRUE(outputLayer != nullptr);
    EXPECT_EQ("ScaleShift", outputLayer->type);

    EXPECT_EQ(1ul, outputLayer->insData.size());
    const InferenceEngine::DataPtr insData = outputLayer->insData[0].lock();
    EXPECT_TRUE(insData != nullptr);
    const InferenceEngine::CNNLayerPtr depthToSpace = getCreatorLayer(insData).lock();
    EXPECT_TRUE(depthToSpace != nullptr);
    EXPECT_EQ("DepthToSpace", depthToSpace->type);

    if (params.updatePrecisions) {
        const InferenceEngine::Precision precision = depthToSpace->outData[0]->getTensorDesc().getPrecision();
        EXPECT_TRUE((precision == InferenceEngine::Precision::U8) || (precision == InferenceEngine::Precision::I8));
    }

    IE_SUPPRESS_DEPRECATED_END
}

void DepthToSpaceTransformation::validateNGraph() {
    // TODO: remove: don't need to validate here

    // InferenceEngine::SizeVector inputShape;
    // InferenceEngine::Precision netPrecision;
    // InferenceEngine::details::LayerTransformation::Params params;
    // LayerTestsUtils::LayerTransformation::LptVersion version;
    // std::tie(netPrecision, inputShape, targetDevice, params, version) = this->GetParam();

    // std::vector<std::shared_ptr<ngraph::Function>> module{ function };
    // ngraph::pass::VisualizeTree("C:\\Projects\\temp\\test.original").run_on_module(module);

    // std::shared_ptr<ngraph::Function> transformedFunction = transformNGraph(params);

    // std::vector<std::shared_ptr<ngraph::Function>> transformedModule{ transformedFunction };
    // ngraph::pass::VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(transformedModule);

    // auto ngPrecision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    // auto input = std::make_shared<ngraph::opset3::Parameter>(ngPrecision, ngraph::Shape(inputShape));
    // auto depthToSpace = std::make_shared<ngraph::opset3::DepthToSpace>(input, ngraph::opset3::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);
    // auto referenceFunction = std::make_shared<ngraph::Function>(ngraph::NodeVector{ depthToSpace }, ngraph::ParameterVector{ input });

    // auto res = compare_functions(f, f_ref);
    // ASSERT_TRUE(res.first) << res.second;
}

TEST_P(DepthToSpaceTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
