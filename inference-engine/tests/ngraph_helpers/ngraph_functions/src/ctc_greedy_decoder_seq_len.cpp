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
        const ngraph::Output<Node>& sequenceLengthData,
        int32_t blankIndex,
        bool mergeRepeated,
        const element::Type& idxPrecision) {
    const auto blankIndexNode = [&] {
        if (idxPrecision == element::i32) {
            const auto blankIdxDataI32 = std::vector<int32_t>{blankIndex};
            return makeConstant(idxPrecision, {1}, blankIdxDataI32);
        } else if (idxPrecision == element::i64) {
            const auto blankIdxDataI64 = std::vector<int64_t>{blankIndex};
            return makeConstant(idxPrecision, {1}, blankIdxDataI64);
        }
        throw std::logic_error("Unsupported index precision");
    }();

    return std::make_shared<op::v6::CTCGreedyDecoderSeqLen>(inputData,
                                                            sequenceLengthData,
                                                            blankIndexNode,
                                                            mergeRepeated,
                                                            idxPrecision,
                                                            idxPrecision);
}

}  // namespace builder
}  // namespace ngraph
