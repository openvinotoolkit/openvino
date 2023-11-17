// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "shared_test_classes/single_layer/logical.hpp"

using namespace LayerTestsDefinitions::LogicalParams;

namespace LayerTestsDefinitions {
std::string LogicalLayerTest::getTestCaseName(const testing::TestParamInfo<LogicalTestParams>& obj) {
    InputShapesTuple inputShapes;
    ngraph::helpers::LogicalTypes comparisonOpType;
    ngraph::helpers::InputLayerType secondInputType;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::string targetName;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShapes, comparisonOpType, secondInputType, netPrecision, inPrc, outPrc, inLayout, outLayout, targetName, additional_config)
        = obj.param;
    std::ostringstream results;

    results << "IS0=" << ov::test::utils::vec2str(inputShapes.first) << "_";
    results << "IS1=" << ov::test::utils::vec2str(inputShapes.second) << "_";
    results << "comparisonOpType=" << comparisonOpType << "_";
    results << "secondInputType=" << secondInputType << "_";
    results << "netPRC=" << netPrecision.name() << "_";
    results << "inPRC=" << inPrc.name() << "_";
    results << "outPRC=" << outPrc.name() << "_";
    results << "inL=" << inLayout << "_";
    results << "outL=" << outLayout << "_";
    results << "trgDev=" << targetName;
    return results.str();
}

std::vector<InputShapesTuple> LogicalLayerTest::combineShapes(const std::map<std::vector<size_t>, std::vector<std::vector<size_t >>>& inputShapes) {
    std::vector<InputShapesTuple> resVec;
    for (auto& inputShape : inputShapes) {
        for (auto& item : inputShape.second) {
            resVec.push_back({inputShape.first, item});
        }

        if (inputShape.second.empty()) {
            resVec.push_back({inputShape.first, {}});
        }
    }
    return resVec;
}

InferenceEngine::Blob::Ptr LogicalLayerTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 2, 0);
}

void LogicalLayerTest::SetupParams() {
    std::tie(inputShapes, logicalOpType, secondInputType, netPrecision,
             inPrc, outPrc, inLayout, outLayout, targetDevice, additional_config) =
        this->GetParam();

    configuration.insert(additional_config.begin(), additional_config.end());
}

void LogicalLayerTest::SetUp() {
    SetupParams();

    auto ngInputsPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);
    ov::ParameterVector inputs {std::make_shared<ov::op::v0::Parameter>(ngInputsPrc, ov::Shape(inputShapes.first))};

    std::shared_ptr<ngraph::Node> logicalNode;
    if (logicalOpType != ngraph::helpers::LogicalTypes::LOGICAL_NOT) {
        OPENVINO_SUPPRESS_DEPRECATED_START
        auto secondInput = ngraph::builder::makeInputLayer(ngInputsPrc, secondInputType, inputShapes.second);
        OPENVINO_SUPPRESS_DEPRECATED_END
        if (secondInputType == ngraph::helpers::InputLayerType::PARAMETER) {
            inputs.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(secondInput));
        }
        logicalNode = ngraph::builder::makeLogical(inputs[0], secondInput, logicalOpType);
    } else {
        logicalNode = ngraph::builder::makeLogical(inputs[0], ngraph::Output<ngraph::Node>(), logicalOpType);
    }

    function = std::make_shared<ngraph::Function>(logicalNode, inputs, "Logical");
}
} // namespace LayerTestsDefinitions
