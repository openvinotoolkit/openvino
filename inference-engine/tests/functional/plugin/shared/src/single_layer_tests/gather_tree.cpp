// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "single_layer_tests/gather_tree.hpp"

namespace LayerTestsDefinitions {
std::string GatherTreeLayerTest::getTestCaseName(const testing::TestParamInfo<GatherTreeParamsTuple> &obj) {
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    ngraph::helpers::InputLayerType secondaryInputType;
    std::string targetName;

    std::tie(inputShape, secondaryInputType, netPrecision, targetName) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "secondaryInputType=" << secondaryInputType << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetName;
    return result.str();
}

void GatherTreeLayerTest::SetUp() {
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    ngraph::helpers::InputLayerType secondaryInputType;

    std::tie(inputShape, secondaryInputType, netPrecision, targetDevice) = GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    std::shared_ptr<ngraph::Node> inp2;
    std::shared_ptr<ngraph::Node> inp3;
    std::shared_ptr<ngraph::Node> inp4;

    auto paramsIn = ngraph::builder::makeParams(ngPrc, { inputShape });
    if (ngraph::helpers::InputLayerType::PARAMETER == secondaryInputType) {
        auto paramsSecond = ngraph::builder::makeParams(ngPrc, { inputShape, {inputShape.at(1)}, {}});
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

TEST_P(GatherTreeLayerTest, CompareWithRefs) {
    Run();
};

} // namespace LayerTestsDefinitions