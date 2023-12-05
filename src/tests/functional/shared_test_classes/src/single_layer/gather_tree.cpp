// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/gather_tree.hpp"

namespace LayerTestsDefinitions {
std::string GatherTreeLayerTest::getTestCaseName(const testing::TestParamInfo<GatherTreeParamsTuple> &obj) {
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    ngraph::helpers::InputLayerType secondaryInputType;
    std::string targetName;

    std::tie(inputShape, secondaryInputType, netPrecision, inPrc, outPrc, inLayout, outLayout, targetName) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "secondaryInputType=" << secondaryInputType << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetName;
    return result.str();
}

void GatherTreeLayerTest::SetUp() {
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    ngraph::helpers::InputLayerType secondaryInputType;

    std::tie(inputShape, secondaryInputType, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice) = GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    std::shared_ptr<ngraph::Node> inp2;
    std::shared_ptr<ngraph::Node> inp3;
    std::shared_ptr<ngraph::Node> inp4;

    ov::ParameterVector paramsIn {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    if (ngraph::helpers::InputLayerType::PARAMETER == secondaryInputType) {
        ov::ParameterVector paramsSecond{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape)),
                                         std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{inputShape.at(1)}),
                                         std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape())};
        paramsIn.insert(paramsIn.end(), paramsSecond.begin(), paramsSecond.end());

        inp2 = paramsIn.at(1);
        inp3 = paramsIn.at(2);
        inp4 = paramsIn.at(3);
    } else if (ngraph::helpers::InputLayerType::CONSTANT == secondaryInputType) {
        auto maxBeamIndex = inputShape.at(2) - 1;

        inp2 = ngraph::builder::makeConstant<float>(ngPrc, inputShape, {}, true, maxBeamIndex);
        inp3 = ngraph::builder::makeConstant<float>(ngPrc, {inputShape.at(1)}, {}, true, maxBeamIndex);
        inp4 = ngraph::builder::makeConstant<float>(ngPrc, {}, {}, true, maxBeamIndex);
    } else {
        throw std::runtime_error("Unsupported inputType");
    }

    auto operationResult = std::make_shared<ngraph::opset4::GatherTree>(paramsIn.front(), inp2, inp3, inp4);

    ngraph::ResultVector results{std::make_shared<ngraph::opset4::Result>(operationResult)};
    function = std::make_shared<ngraph::Function>(results, paramsIn, "GatherTree");
}

InferenceEngine::Blob::Ptr GatherTreeLayerTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    auto& shape = function->get_parameters()[0]->get_output_shape(0);
    auto& vecDims = info.getTensorDesc().getDims();

    auto maxBeamIndx = shape.at(2) - 1;

    if (vecDims.size() == 1 || vecDims.size() == 0) { //max_seq_len vector || end_token
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), maxBeamIndx, maxBeamIndx / 2);
    }

    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), maxBeamIndx);
}
} // namespace LayerTestsDefinitions
