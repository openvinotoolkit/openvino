// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "shared_test_classes/single_layer/ctc_loss.hpp"

namespace LayerTestsDefinitions {

std::string CTCLossLayerTest::getTestCaseName(const testing::TestParamInfo<CTCLossParams>& obj) {
    InferenceEngine::SizeVector logitsShapes;
    InferenceEngine::Precision fpPrecision, intPrecision;
    bool preprocessCollapseRepeated, ctcMergeRepeated, unique;
    std::vector<int> logitsLength, labelsLength;
    std::vector<std::vector<int>> labels;
    int blankIndex;
    std::string targetDevice;
    CTCLossParamsSubset ctcLossArgsSubset;
    std::tie(ctcLossArgsSubset, fpPrecision, intPrecision, targetDevice) = obj.param;
    std::tie(logitsShapes, logitsLength, labels, labelsLength, blankIndex, preprocessCollapseRepeated,
        ctcMergeRepeated, unique) = ctcLossArgsSubset;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(logitsShapes) << "_";
    result << "LL=" << ov::test::utils::vec2str(logitsLength) << "_";
    result << "A=" << ov::test::utils::vec2str(labels) << "_";
    result << "AL=" << ov::test::utils::vec2str(labelsLength) << "_";
    result << "BI=" << blankIndex << "_";
    result << "PCR=" << preprocessCollapseRepeated << "_";
    result << "CMR=" << ctcMergeRepeated << "_";
    result << "U=" << unique << "_";
    result << "PF=" << fpPrecision.name() << "_";
    result << "PI=" << intPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void CTCLossLayerTest::SetUp() {
    std::vector<size_t> logitsShapes;
    InferenceEngine::Precision fpPrecision, intPrecision;
    bool preprocessCollapseRepeated, ctcMergeRepeated, unique;
    std::vector<int> logitsLength, labelsLength;
    std::vector<std::vector<int>> labels;
    int blankIndex;
    CTCLossParamsSubset ctcLossArgsSubset;
    std::tie(ctcLossArgsSubset, fpPrecision, intPrecision, targetDevice) = this->GetParam();
    std::tie(logitsShapes, logitsLength, labels, labelsLength, blankIndex, preprocessCollapseRepeated,
        ctcMergeRepeated, unique) = ctcLossArgsSubset;

    auto ngFpPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(fpPrecision);
    auto ngIntPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(intPrecision);

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngFpPrc, ov::Shape(logitsShapes))};
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto ctcLoss = std::dynamic_pointer_cast<ngraph::opset4::CTCLoss>(
            ngraph::builder::makeCTCLoss(paramOuts[0], logitsLength, labels, labelsLength, blankIndex,
                ngFpPrc, ngIntPrc, preprocessCollapseRepeated, ctcMergeRepeated, unique));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(ctcLoss)};
    function = std::make_shared<ngraph::Function>(results, params, "CTCLoss");
}
}  // namespace LayerTestsDefinitions
