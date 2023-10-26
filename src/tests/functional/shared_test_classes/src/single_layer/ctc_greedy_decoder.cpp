// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/ctc_greedy_decoder.hpp"
#include "ov_models/builders.hpp"

namespace LayerTestsDefinitions {
std::string CTCGreedyDecoderLayerTest::getTestCaseName(
        const testing::TestParamInfo<ctcGreedyDecoderParams>& obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    bool mergeRepeated;
    std::tie(netPrecision,
        inPrc, outPrc, inLayout, outLayout,
        inputShapes,
        mergeRepeated,
        targetDevice) = obj.param;

    std::ostringstream result;
    const char separator = '_';

    result << "IS="     << ov::test::utils::vec2str(inputShapes) << separator;
    result << "netPRC=" << netPrecision.name() << separator;
    result << "inPRC=" << inPrc.name() << separator;
    result << "outPRC=" << outPrc.name() << separator;
    result << "inL=" << inLayout << separator;
    result << "outL=" << outLayout << separator;
    result << "merge_repeated=" << std::boolalpha << mergeRepeated << separator;
    result << "trgDev=" << targetDevice;

    return result.str();
}

void CTCGreedyDecoderLayerTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes;
    bool mergeRepeated;
    std::tie(netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, mergeRepeated, targetDevice) = GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector paramsIn {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes))};
    auto paramOuts = ngraph::helpers::convert2OutputVector(
        ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramsIn));

    auto ctcGreedyDecoder = std::dynamic_pointer_cast<ngraph::opset1::CTCGreedyDecoder>(
            ngraph::builder::makeCTCGreedyDecoder(paramOuts[0], mergeRepeated));

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(ctcGreedyDecoder) };
    function = std::make_shared<ngraph::Function>(results, paramsIn, "CTCGreedyDecoder");
}
}  // namespace LayerTestsDefinitions
