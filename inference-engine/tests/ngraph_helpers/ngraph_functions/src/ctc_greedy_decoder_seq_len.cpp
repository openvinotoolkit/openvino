// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeCTCGreedyDecoderSeqLen(
        const ngraph::Output<Node>& inputData,
        int blankIndex,
        bool mergeRepeated,
        const element::Type& idxPrec) {
    const auto& inputDataShape = inputData.get_shape();
    const size_t B = inputDataShape[0];
    const size_t T = inputDataShape[1];

    std::mt19937 gen(1);
    std::uniform_int_distribution<unsigned long> dist(0, T);

    std::vector<int> sequenceLenData(B);
    for (int b = 0; b < B; b++) {
        int len = dist(gen);
        sequenceLenData[b] = len;
    }

    auto sequenceLenNode = makeConstant(idxPrec, {B}, sequenceLenData);

    std::vector<int> blankIdxData = {blankIndex};
    auto blankIndexNode = makeConstant(idxPrec, {1}, blankIdxData);

    return std::make_shared<op::v6::CTCGreedyDecoderSeqLen>(inputData, sequenceLenNode, blankIndexNode, mergeRepeated, idxPrec, idxPrec);
}
}  // namespace builder
}  // namespace ngraph
