// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/variadic_split.hpp"

namespace LayerTestsDefinitions {

    std::string VariadicSplitLayerTest::getTestCaseName(const testing::TestParamInfo<VariadicSplitParams>& obj) {
        int64_t axis;
        std::vector<size_t> numSplits;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::Precision inPrc, outPrc;
        InferenceEngine::Layout inLayout, outLayout;
        InferenceEngine::SizeVector inputShapes;
        std::string targetDevice;
        std::tie(numSplits, axis, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetDevice) = obj.param;
        std::ostringstream result;
        result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
        result << "numSplits=" << ov::test::utils::vec2str(numSplits) << "_";
        result << "axis=" << axis << "_";
        result << "IS";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "inPRC=" << inPrc.name() << "_";
        result << "outPRC=" << outPrc.name() << "_";
        result << "inL=" << inLayout << "_";
        result << "outL=" << outLayout << "_";
        result << "trgDev=" << targetDevice;
        return result.str();
    }

    void VariadicSplitLayerTest::SetUp() {
        int64_t axis;
        std::vector<size_t> inputShape, numSplits;
        InferenceEngine::Precision netPrecision;
        std::tie(numSplits, axis, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
        auto VariadicSplit = std::dynamic_pointer_cast<ngraph::opset3::VariadicSplit>(ngraph::builder::makeVariadicSplit(params[0], numSplits,
                axis));
        ngraph::ResultVector results;
        for (int i = 0; i < numSplits.size(); i++) {
            results.push_back(std::make_shared<ngraph::opset3::Result>(VariadicSplit->output(i)));
        }
        function = std::make_shared<ngraph::Function>(results, params, "VariadicSplit");
    }

}  // namespace LayerTestsDefinitions
