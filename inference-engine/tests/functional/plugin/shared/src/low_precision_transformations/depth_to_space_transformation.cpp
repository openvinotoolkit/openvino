// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/depth_to_space_transformation.hpp"

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

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <legacy/ngraph_ops/fully_connected.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/common_optimizations/depth_to_space_fusion.hpp>
#include <ngraph/op/depth_to_space.hpp>

namespace LayerTestsDefinitions {

std::string DepthToSpaceTransformation::getTestCaseName(testing::TestParamInfo<LayerTestsUtils::LayerTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, params) = obj.param;

    std::ostringstream result;
    result << netPrecision.name() << "_" << targetDevice << "_" << toString(params);
    return result.str();
}

void DepthToSpaceTransformation::SetUp() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = this->GetParam();
    if (inputShape.size() != 4ul) {
        THROW_IE_EXCEPTION << "not supported input shape size " << inputShape.size();
    }

    auto ngPrecision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    const auto input = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, ngraph::Shape(inputShape));

    const auto fakeQuantize = ngraph::builder::makeFakeQuantize(input, ngPrecision, 256ul, { 1ul });

    const auto shapeReshapeBefore = ngraph::opset1::Constant::create(
        ngraph::element::i64,
        ngraph::Shape{ 6ul },
        ngraph::Shape{ inputShape[0], inputShape[1] / 4ul, 2ul, 2ul, inputShape[2], inputShape[3] });
    const auto reshapeBefore = std::make_shared<ngraph::opset1::Reshape>(fakeQuantize, shapeReshapeBefore, false);

    const auto permutation = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 6 }, { 0, 1, 4, 2, 5, 3 });
    const auto permute = std::make_shared<ngraph::opset1::Transpose>(reshapeBefore, permutation);

    const auto shapeReshapeAfter = ngraph::opset1::Constant::create(
        ngraph::element::i64,
        ngraph::Shape{ 4 },
        ngraph::Shape{ 1, inputShape[1] / 4ul, inputShape[2] * 2, inputShape[3] * 2 });
    const auto reshapeAfter = std::make_shared<ngraph::opset1::Reshape>(permute, shapeReshapeAfter, false);

    function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ reshapeAfter }, ngraph::ParameterVector{ input });

    ngraph::pass::InitNodeInfo().run_on_function(function);
    ngraph::pass::DepthToSpaceFusion().run_on_function(function);

    // TODO: move to some another place
    validate();
}

void DepthToSpaceTransformation::validate() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = this->GetParam();

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

TEST_P(DepthToSpaceTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
