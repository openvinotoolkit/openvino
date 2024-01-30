// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/ctc_greedy_decoder_seq_len.hpp"

#include <memory>
#include <vector>

#include "common_test_utils/node_builders/constant.hpp"
#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ov::Node> makeCTCGreedyDecoderSeqLen(const ov::Output<ov::Node>& inputData,
                                                     const ov::Output<ov::Node>& sequenceLengthData,
                                                     int32_t blankIndex,
                                                     bool mergeRepeated,
                                                     const ov::element::Type& idxPrecision) {
    const auto blankIndexNode = [&] {
        if (idxPrecision == ov::element::i32) {
            const auto blankIdxDataI32 = std::vector<int32_t>{blankIndex};
            return ov::test::utils::deprecated::make_constant(idxPrecision, {1}, blankIdxDataI32);
        } else if (idxPrecision == ov::element::i64) {
            const auto blankIdxDataI64 = std::vector<int64_t>{blankIndex};
            return ov::test::utils::deprecated::make_constant(idxPrecision, {1}, blankIdxDataI64);
        }
        throw std::logic_error("Unsupported index precision");
    }();

    return std::make_shared<ov::op::v6::CTCGreedyDecoderSeqLen>(inputData,
                                                                sequenceLengthData,
                                                                blankIndexNode,
                                                                mergeRepeated,
                                                                idxPrecision,
                                                                idxPrecision);
}

std::shared_ptr<ov::Node> makeCTCGreedyDecoderSeqLen(const ov::Output<ov::Node>& inputData,
                                                     int32_t blankIndex,
                                                     bool mergeRepeated,
                                                     const ov::element::Type& idxPrecision) {
    const auto sequenceLengthData = [&] {
        const size_t N = inputData.get_shape().at(0);
        const size_t T = inputData.get_shape().at(1);

        if (idxPrecision == ov::element::i32) {
            const auto sequenceLengthI32 = std::vector<int32_t>(N, T);
            return ov::test::utils::deprecated::make_constant(idxPrecision, {N}, sequenceLengthI32);
        } else if (idxPrecision == ov::element::i64) {
            const auto sequenceLengthI64 = std::vector<int64_t>(N, T);
            return ov::test::utils::deprecated::make_constant(idxPrecision, {N}, sequenceLengthI64);
        }
        throw std::logic_error("Unsupported index precision");
    }();

    return makeCTCGreedyDecoderSeqLen(inputData, sequenceLengthData, blankIndex, mergeRepeated, idxPrecision);
}

}  // namespace builder
}  // namespace ngraph
