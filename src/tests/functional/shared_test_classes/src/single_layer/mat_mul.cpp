// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "shared_test_classes/single_layer/mat_mul.hpp"

namespace LayerTestsDefinitions {

std::vector<ShapeRelatedParams> MatMulTest::combineShapes(const std::vector<std::vector<size_t>>& firstInputShapes,
                                                          const std::vector<std::vector<size_t>>& secondInputShapes,
                                                          bool transposeA,
                                                          bool transposeB) {
    std::vector<ShapeRelatedParams> resVec;
    for (const auto& firstInputShape : firstInputShapes) {
        for (const auto& secondInputShape : secondInputShapes) {
            resVec.push_back(ShapeRelatedParams{ {firstInputShape, transposeA}, {secondInputShape, transposeB } });
        }
    }
    return resVec;
}

std::string MatMulTest::getTestCaseName(const testing::TestParamInfo<MatMulLayerTestParamsSet> &obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout;
    ShapeRelatedParams shapeRelatedParams;
    ngraph::helpers::InputLayerType secondaryInputType;
    std::string targetDevice;
    std::map<std::string, std::string> additionalConfig;
    std::tie(shapeRelatedParams, netPrecision, inPrc, outPrc, inLayout, secondaryInputType, targetDevice, additionalConfig) =
        obj.param;

    std::ostringstream result;
    result << "IS0=" << ov::test::utils::vec2str(shapeRelatedParams.input1.first) << "_";
    result << "IS1=" << ov::test::utils::vec2str(shapeRelatedParams.input2.first) << "_";
    result << "transpose_a=" << shapeRelatedParams.input1.second << "_";
    result << "transpose_b=" << shapeRelatedParams.input2.second << "_";
    result << "secondaryInputType=" << secondaryInputType << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "trgDev=" << targetDevice;
    result << "config=(";
    for (const auto& configEntry : additionalConfig) {
        result << configEntry.first << ", " << configEntry.second << ";";
    }
    result << ")";
    return result.str();
}

void MatMulTest::SetUp() {
    ShapeRelatedParams shapeRelatedParams;
    ngraph::helpers::InputLayerType secondaryInputType;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::map<std::string, std::string> additionalConfig;
    std::tie(shapeRelatedParams, netPrecision, inPrc, outPrc, inLayout, secondaryInputType, targetDevice, additionalConfig) =
        this->GetParam();

    configuration.insert(additionalConfig.begin(), additionalConfig.end());

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(shapeRelatedParams.input1.first))};

    auto secondaryInput = ngraph::builder::makeInputLayer(ngPrc, secondaryInputType, shapeRelatedParams.input2.first);
    if (secondaryInputType == ngraph::helpers::InputLayerType::PARAMETER) {
        params.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(secondaryInput));
    }
    auto MatMul = std::dynamic_pointer_cast<ngraph::opset3::MatMul>(
            ngraph::builder::makeMatMul(params[0], secondaryInput, shapeRelatedParams.input1.second, shapeRelatedParams.input2.second));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(MatMul)};
    function = std::make_shared<ngraph::Function>(results, params, "MatMul");
}

}  // namespace LayerTestsDefinitions
