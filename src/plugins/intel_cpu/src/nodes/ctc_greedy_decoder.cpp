// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/ctc_greedy_decoder.hpp"

#include <string>
#include <vector>

#include "ctc_greedy_decoder.h"
#include "openvino/core/parallel.hpp"

namespace ov::intel_cpu::node {

bool CTCGreedyDecoder::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                            std::string& errorMessage) noexcept {
    try {
        const auto greedyDecOp = ov::as_type_ptr<const ov::op::v0::CTCGreedyDecoder>(op);
        if (!greedyDecOp) {
            errorMessage = "Node is not an instance of the CTCGreedyDecoder operation from operation set v0.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

CTCGreedyDecoder::CTCGreedyDecoder(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    if (getOriginalInputsNumber() != 2) {
        THROW_CPU_NODE_ERR("has invalid number of input edges: ", getOriginalInputsNumber());
    }
    if (getOriginalOutputsNumber() != 1) {
        THROW_CPU_NODE_ERR("has invalid number of outputs edges: ", getOriginalOutputsNumber());
    }

    const auto& dataDims = getInputShapeAtPort(DATA_INDEX).getDims();
    const auto& seqDims = getInputShapeAtPort(SEQUENCE_LENGTH_INDEX).getDims();

    if (!dimsEqualWeak(dataDims[0], seqDims[0]) || !dimsEqualWeak(dataDims[1], seqDims[1])) {
        THROW_CPU_NODE_ERR("has invalid input shapes.");
    }

    auto greedyDecOp = ov::as_type_ptr<const ov::op::v0::CTCGreedyDecoder>(op);
    mergeRepeated = greedyDecOp->get_ctc_merge_repeated();
}

void CTCGreedyDecoder::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    ov::element::Type inDataPrecision = getOriginalInputPrecisionAtPort(DATA_INDEX);
    if (!one_of(inDataPrecision, ov::element::f32, ov::element::bf16, ov::element::f16)) {
        THROW_CPU_NODE_ERR("has unsupported 'data' input precision: ", inDataPrecision);
    }

    ov::element::Type seqLenPrecision = getOriginalInputPrecisionAtPort(SEQUENCE_LENGTH_INDEX);
    if (!one_of(seqLenPrecision, ov::element::f32, ov::element::bf16, ov::element::f16)) {
        THROW_CPU_NODE_ERR("has unsupported 'sequence_length' input precision: ", seqLenPrecision);
    }

    addSupportedPrimDesc({{LayoutType::ncsp, ov::element::f32}, {LayoutType::ncsp, ov::element::f32}},
                         {{LayoutType::ncsp, ov::element::f32}},
                         impl_desc_type::ref_any);
}

void CTCGreedyDecoder::execute(const dnnl::stream& strm) {
    const auto* probabilities = getSrcDataAtPortAs<const float>(DATA_INDEX);
    const auto* sequenceMask = getSrcDataAtPortAs<const float>(SEQUENCE_LENGTH_INDEX);
    auto* outputSequences = getDstDataAtPortAs<float>(0);

    const size_t T = getParentEdgeAt(DATA_INDEX)->getMemory().getStaticDims()[0];
    const size_t B = getParentEdgeAt(DATA_INDEX)->getMemory().getStaticDims()[1];
    const int C = getParentEdgeAt(DATA_INDEX)->getMemory().getStaticDims()[2];
    const size_t BC = B * C;
    const size_t CB1 = C * (B - 1);

    const int blankIndex = C - 1;

    std::vector<size_t> sequenceLengths(B, 0);
    parallel_for(B, [&](size_t b) {
        size_t t = 0;
        for (; t < T; t++) {
            if (sequenceMask[B * t + b] == 0.f) {
                break;
            }
        }
        sequenceLengths[b] = t;
    });

    size_t workAmount = 0;
    for (size_t b = 0; b < B; b++) {
        workAmount += sequenceLengths[b];
    }

    // Parallelization could not be made directly by T due to output index depends on merged classes and
    // blank index, thus could not be shared between threads. Better to divide operation on two steps.
    // At the first stage find the maximum index. At second stage merge if needed.
    // Such approach makes parallelization more efficient.
    auto threadBody = [&](const int ithr, const int nthr) {
        size_t start(0lu), end(0lu);
        splitter(workAmount, nthr, ithr, start, end);
        if (start >= end) {
            return;
        }
        size_t tStart = 0lu, bStart = 0lu;
        for (; bStart < B; bStart++) {
            tStart += sequenceLengths[bStart];
            if (tStart >= start) {
                tStart = start - (tStart - sequenceLengths[bStart]);
                break;
            }
        }

        size_t workCounter = start;

        for (size_t b = bStart; b < B; ++b) {
            size_t outputIndex = b * T + tStart;
            const float* probs = probabilities + b * C + BC * tStart;
            size_t sequenceLength = sequenceLengths[b];

            for (size_t t = tStart; t < sequenceLength; ++t) {
                int maxClassIdx = 0;

                float maxProb = probs[0];
                ++probs;

                for (int c = 1; c < C; ++c, ++probs) {
                    if (*probs > maxProb) {
                        maxClassIdx = c;
                        maxProb = *probs;
                    }
                }
                probs += CB1;
                outputSequences[outputIndex++] = static_cast<float>(maxClassIdx);

                if (++workCounter >= end) {
                    return;
                }
            }
            tStart = 0lu;
        }
    };  // thread body

    parallel_nt(0, threadBody);

    parallel_for(B, [&](size_t b) {
        int prevClassIdx = -1;
        size_t outputIndex = b * T;
        const size_t sequenceLength = sequenceLengths[b];
        float* shiftedOut = outputSequences + b * T;
        for (size_t t = 0; t < sequenceLength; ++t) {
            if (*shiftedOut < blankIndex && !(mergeRepeated && *shiftedOut == prevClassIdx)) {
                outputSequences[outputIndex++] = *shiftedOut;
            }
            prevClassIdx = *shiftedOut;
            shiftedOut++;
        }
        std::fill(outputSequences + outputIndex, outputSequences + (b + 1) * T, -1.f);
    });
}

bool CTCGreedyDecoder::created() const {
    return getType() == Type::CTCGreedyDecoder;
}

void CTCGreedyDecoder::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool CTCGreedyDecoder::needPrepareParams() const {
    return false;
}

}  // namespace ov::intel_cpu::node
