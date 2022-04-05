// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeCTCGreedyDecoder(
        const ngraph::Output<Node>& inputData,
        const bool mergeRepeated) {
    auto inputDataShape = inputData.get_shape();
    size_t T = inputDataShape[0];
    size_t B = inputDataShape[1];

    std::mt19937 gen(1);
    std::uniform_int_distribution<unsigned long> dist(1, T);

    std::vector<int> sequenceMaskData(B * T, 0);
    for (int b = 0; b < B; b++) {
        int len = dist(gen);
        for (int t = 0; t < len; t++) {
            sequenceMaskData[t * B + b] = 1;
        }
    }

    auto sequenceMaskNode = makeConstant(inputData.get_element_type(), {T, B}, sequenceMaskData);

    auto CTCGreedyDecoderNode = std::make_shared<opset1::CTCGreedyDecoder>(inputData, sequenceMaskNode, mergeRepeated);

    return CTCGreedyDecoderNode;
}
}  // namespace builder
}  // namespace ngraph
