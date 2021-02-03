// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/single_layer/ctc_greedy_decoder_seq_len.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {
std::string CTCGreedyDecoderSeqLenLayerTest::getTestCaseName(
        const testing::TestParamInfo<ctcGreedyDecoderSeqLenParams>& obj) {
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision dataPrecision, indicesPrecision;
    int blankIndex;
    bool mergeRepeated;
    std::string targetDevice;
    std::tie(inputShapes,
        dataPrecision,
        indicesPrecision,
        blankIndex,
        mergeRepeated,
        targetDevice) = obj.param;

    std::ostringstream result;

    result << "IS="     << CommonTestUtils::vec2str(inputShapes) << '_';
    result << "dataPRC=" << dataPrecision.name() << '_';
    result << "idxPRC=" << indicesPrecision.name() << '_';
    result << "BlankIdx=" << blankIndex << '_';
    result << "mergeRepeated=" << std::boolalpha << mergeRepeated << '_';
    result << "trgDev=" << targetDevice;

    return result.str();
}

void CTCGreedyDecoderSeqLenLayerTest::SetUp() {
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision dataPrecision, indicesPrecision;
    int blankIndex;
    bool mergeRepeated;
    std::tie(inputShapes,
        dataPrecision,
        indicesPrecision,
        blankIndex,
        mergeRepeated,
        targetDevice) = GetParam();

    auto ngDataPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(dataPrecision);
    auto ngIdxPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(indicesPrecision);
    auto paramsIn = ngraph::builder::makeParams(ngDataPrc, { inputShapes });
    auto paramOuts = ngraph::helpers::convert2OutputVector(
        ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramsIn));

    auto ctcGreedyDecoderSeqLen = std::dynamic_pointer_cast<ngraph::op::v6::CTCGreedyDecoderSeqLen>(
            ngraph::builder::makeCTCGreedyDecoderSeqLen(paramOuts[0], blankIndex, mergeRepeated, ngIdxPrc));

    ngraph::ResultVector results;
    for (int i = 0; i < ctcGreedyDecoderSeqLen->get_output_size(); i++) {
        results.push_back(std::make_shared<ngraph::opset1::Result>(ctcGreedyDecoderSeqLen->output(i)));
    }
    function = std::make_shared<ngraph::Function>(results, paramsIn, "CTCGreedyDecoderSeqLen");
}
}  // namespace LayerTestsDefinitions
