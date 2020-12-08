// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <functional>
#include <cmath>
#include <memory>
#include <functional_test_utils/skip_tests_config.hpp>

#include "ie_core.hpp"
#include "ie_precision.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "single_layer_tests/ctc_greedy_decoder.hpp"

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

    result << "IS="     << CommonTestUtils::vec2str(inputShapes) << separator;
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
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, mergeRepeated, targetDevice) = GetParam();
    sequenceLengths = { inputShapes.at(0), inputShapes.at(1) };
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto paramsIn = ngraph::builder::makeParams(ngPrc, { inputShapes, sequenceLengths });
    auto paramsOut = ngraph::helpers::convert2OutputVector(
        ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramsIn));
    auto ctcGreedyDecoder = std::make_shared<ngraph::opset1::CTCGreedyDecoder>(
        paramsOut[0],
        paramsOut[1],
        mergeRepeated);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(ctcGreedyDecoder) };
    function = std::make_shared<ngraph::Function>(results, paramsIn, "Grn");
}

TEST_P(CTCGreedyDecoderLayerTest, CompareWithRefs) {
    Run();
};
}  // namespace LayerTestsDefinitions
