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
    std::string targetName;

    std::tie(inputShape, netPrecision, targetName) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetName;
    return result.str();
}

void GatherTreeLayerTest::SetUp() {
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;

    std::tie(inputShape, netPrecision, targetDevice) = GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto paramsIn = ngraph::builder::makeParams(ngPrc, { inputShape , inputShape, {inputShape.at(BATCH_SIZE)}, {}});

    auto operationResult = std::make_shared<ngraph::opset4::GatherTree>(paramsIn[0], paramsIn[1], paramsIn[2], paramsIn[3]);

    ngraph::ResultVector results{std::make_shared<ngraph::opset4::Result>(operationResult)};
    function = std::make_shared<ngraph::Function>(results, paramsIn, "GatherTree");
}

InferenceEngine::Blob::Ptr GatherTreeLayerTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    auto& shape = function->get_parameters()[0]->get_output_shape(0);
    auto& vecDims = info.getTensorDesc().getDims();

    auto maxBeamIndx = shape.at(BEAM_WIDTH) - 1;

    if (vecDims.size() == 1 || vecDims.size() == 0) { //max_seq_len vector || end_token
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), maxBeamIndx, maxBeamIndx / 2);
    }

    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), maxBeamIndx);
}

TEST_P(GatherTreeLayerTest, CompareWithRefs) {
    Run();
};

} // namespace LayerTestsDefinitions