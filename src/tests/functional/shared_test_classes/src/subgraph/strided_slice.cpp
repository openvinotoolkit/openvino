// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"

#include "shared_test_classes/subgraph/strided_slice.hpp"

namespace SubgraphTestsDefinitions {

std::string StridedSliceTest::getTestCaseName(const testing::TestParamInfo<StridedSliceParams> &obj) {
    StridedSliceSpecificParams params;
    InferenceEngine::Precision netPrc;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::string targetName;
    std::map<std::string, std::string> additionalConfig;
    std::tie(params, netPrc, inPrc, outPrc, inLayout, outLayout, targetName, additionalConfig) = obj.param;
    std::ostringstream result;
    result << "inShape=" << ov::test::utils::vec2str(params.inputShape) << "_";
    result << "netPRC=" << netPrc.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "begin=" << ov::test::utils::vec2str(params.begin) << "_";
    result << "end=" << ov::test::utils::vec2str(params.end) << "_";
    result << "stride=" << ov::test::utils::vec2str(params.strides) << "_";
    result << "begin_m=" << ov::test::utils::vec2str(params.beginMask) << "_";
    result << "end_m=" << ov::test::utils::vec2str(params.endMask) << "_";
    result << "new_axis_m=" << (params.newAxisMask.empty() ? "def" : ov::test::utils::vec2str(params.newAxisMask)) << "_";
    result << "shrink_m=" << (params.shrinkAxisMask.empty() ? "def" : ov::test::utils::vec2str(params.shrinkAxisMask)) << "_";
    result << "ellipsis_m=" << (params.ellipsisAxisMask.empty() ? "def" : ov::test::utils::vec2str(params.ellipsisAxisMask)) << "_";
    result << "trgDev=" << targetName;
    for (auto const& configItem : additionalConfig) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void StridedSliceTest::SetUp() {
    StridedSliceSpecificParams ssParams;
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> additionalConfig;
    std::tie(ssParams, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice, additionalConfig) = this->GetParam();
    configuration.insert(additionalConfig.begin(), additionalConfig.end());

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(ssParams.inputShape))};
    auto relu = std::make_shared<ngraph::opset1::Relu>(params[0]);

    ov::Shape constShape = {ssParams.begin.size()};
    auto beginNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, constShape, ssParams.begin.data());
    auto endNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, constShape, ssParams.end.data());
    auto strideNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, constShape, ssParams.strides.data());

    auto ss = std::make_shared<ov::op::v1::StridedSlice>(relu,
                                                        beginNode,
                                                        endNode,
                                                        strideNode,
                                                        ssParams.beginMask,
                                                        ssParams.endMask,
                                                        ssParams.newAxisMask,
                                                        ssParams.shrinkAxisMask,
                                                        ssParams.ellipsisAxisMask);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(ss)};
    function = std::make_shared<ngraph::Function>(results, params, "strided_slice");
}

}  // namespace SubgraphTestsDefinitions
