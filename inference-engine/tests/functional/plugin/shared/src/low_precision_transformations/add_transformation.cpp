// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/add_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ngraph_functions/low_precision_transformations/add_function.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

namespace LayerTestsDefinitions {

std::string AddTransformation::getTestCaseName(testing::TestParamInfo< AddTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params = LayerTestsUtils::LayerTransformationParamsFactory::createParams();
    LayerTestsUtils::LayerTransformation::LptVersion version;
    AddTestValues param;
    std::tie(netPrecision, inputShapes, targetDevice, version, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params, version) <<
        param.precisionOnActivations <<
        (param.broadcast ? "_broadcast" : "");
    if (!param.fakeQuantize1.empty()) {
        result << "_on_branch1_" <<
            param.fakeQuantize1.inputLowValues[0] << "_" <<
            param.fakeQuantize1.inputHighValues[0] << "_" <<
            param.fakeQuantize1.outputLowValues[0] << "_" <<
            param.fakeQuantize1.outputHighValues[0];
    }
    if (!param.fakeQuantize2.empty()) {
        result << "_on_branch2_" <<
            param.fakeQuantize2.inputLowValues[0] << "_" <<
            param.fakeQuantize2.inputHighValues[0] << "_" <<
            param.fakeQuantize2.outputLowValues[0] << "_" <<
            param.fakeQuantize2.outputHighValues[0];
    }
    return result.str();
}

void AddTransformation::SetUp() {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShape1;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    AddTestValues param;
    std::tie(netPrecision, inputShape1, targetDevice, version, param) = this->GetParam();
    auto precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    InferenceEngine::SizeVector inputShape2 = inputShape1;

    if (param.broadcast) {
        inputShape2[2] = 1;
        inputShape2[3] = 1;
    }

    auto fq1 = param.fakeQuantize1;
    auto fq2 = param.fakeQuantize2;

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape1));
    const auto fakeQuantize1 = fq1.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            input1, precision, fq1.quantizationLevel, fq1.constantShape,
            fq1.inputLowValues, fq1.inputHighValues, fq1.outputLowValues, fq1.outputHighValues);

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    const auto fakeQuantize2 = fq2.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            input2, precision, fq2.quantizationLevel, fq2.constantShape,
            fq2.inputLowValues, fq2.inputHighValues, fq2.outputLowValues, fq2.outputHighValues);

    const auto add = std::make_shared<ngraph::opset1::Add>(
        fq1.empty() ? input1 : fakeQuantize1,
        fq2.empty() ? input2 : fakeQuantize2);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(add) };
    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input1, input2 }, "AddTransformation");

    ngraph::pass::InitNodeInfo().run_on_function(function);

    if (version == LptVersion::cnnNetwork) {
        validate();
    }
}

void AddTransformation::validate() {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShape;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params = LayerTestsUtils::LayerTransformationParamsFactory::createParams();
    LayerTestsUtils::LayerTransformation::LptVersion version;
    AddTestValues param;
    std::tie(netPrecision, inputShape, targetDevice, version, param) = this->GetParam();

    params.precisionsOnActivations = param.precisionOnActivations;

    const InferenceEngine::CNNNetwork network = transform(params);

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());
    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = getCreatorLayer(it->second).lock();
    EXPECT_TRUE(outputLayer != nullptr);

    if (!((param.fakeQuantize1.empty()) && (param.fakeQuantize2.empty()))) {
        EXPECT_EQ("ScaleShift", outputLayer->type);

        EXPECT_EQ(1ul, outputLayer->insData.size());
        const InferenceEngine::DataPtr insData = outputLayer->insData[0].lock();
        EXPECT_TRUE(insData != nullptr);
        const InferenceEngine::CNNLayerPtr layer = getCreatorLayer(insData).lock();
        EXPECT_TRUE(layer != nullptr);
        EXPECT_EQ("Eltwise", layer->type);

        if (params.updatePrecisions) {
            const InferenceEngine::Precision precision1 =
                InferenceEngine::details::CNNNetworkHelper::getParents(*layer)[0]->outData[0]->getPrecision();
            const InferenceEngine::Precision precision2 =
                InferenceEngine::details::CNNNetworkHelper::getParents(*layer)[1]->outData[0]->getPrecision();

            EXPECT_EQ(precision1, param.expectedPrecisions[0]);
            EXPECT_EQ(precision2, param.expectedPrecisions[1]);
        }
    } else {
        EXPECT_EQ("Eltwise", outputLayer->type);
    }
    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(AddTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
