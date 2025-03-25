// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ctc_greedy_decoder_seq_len.h"

#include <openvino/op/ctc_greedy_decoder_seq_len.hpp>
#include <string>
#include <vector>

#include "openvino/core/parallel.hpp"

namespace ov::intel_cpu::node {

bool CTCGreedyDecoderSeqLen::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                                  std::string& errorMessage) noexcept {
    try {
        const auto greedyDecOp = ov::as_type_ptr<const ov::op::v6::CTCGreedyDecoderSeqLen>(op);
        if (!greedyDecOp) {
            errorMessage = "Node is not an instance of the CTCGreedyDecoderSeqLen operation from operation set v6.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

CTCGreedyDecoderSeqLen::CTCGreedyDecoderSeqLen(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    if (getOriginalInputsNumber() < 2 || getOriginalInputsNumber() > 3) {
        THROW_CPU_NODE_ERR("has invalid number of input edges: ", getOriginalInputsNumber());
    }
    if (getOriginalOutputsNumber() != 2) {
        THROW_CPU_NODE_ERR("has invalid number of outputs edges: ", getOriginalOutputsNumber());
    }

    const auto& dataDims = getInputShapeAtPort(DATA_INDEX).getDims();
    const auto& seqDims = getInputShapeAtPort(SEQUENCE_LENGTH_INDEX).getDims();
    if (!dimsEqualWeak(dataDims[0], seqDims[0])) {
        THROW_CPU_NODE_ERR("has invalid input shapes.");
    }

    auto greedyDecOp = ov::as_type_ptr<const ov::op::v6::CTCGreedyDecoderSeqLen>(op);
    mergeRepeated = greedyDecOp->get_merge_repeated();
}

void CTCGreedyDecoderSeqLen::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    ov::element::Type inDataPrecision = getOriginalInputPrecisionAtPort(DATA_INDEX);
    if (!one_of(inDataPrecision, ov::element::f32, ov::element::bf16, ov::element::f16)) {
        THROW_CPU_NODE_ERR("has unsupported 'data' input precision: ", inDataPrecision);
    }

    ov::element::Type seqLenPrecision = getOriginalInputPrecisionAtPort(SEQUENCE_LENGTH_INDEX);
    if (seqLenPrecision != ov::element::i32 && seqLenPrecision != ov::element::i64) {
        THROW_CPU_NODE_ERR("has unsupported 'sequence_length' input precision: ", seqLenPrecision);
    }

    std::vector<PortConfigurator> inDataConf;
    inDataConf.reserve(inputShapes.size());
    inDataConf.emplace_back(LayoutType::ncsp, ov::element::f32);
    for (size_t i = 1; i < inputShapes.size(); ++i) {
        inDataConf.emplace_back(LayoutType::ncsp, ov::element::i32);
    }

    addSupportedPrimDesc(inDataConf,
                         {{LayoutType::ncsp, ov::element::i32}, {LayoutType::ncsp, ov::element::i32}},
                         impl_desc_type::ref_any);
}

void CTCGreedyDecoderSeqLen::execute(const dnnl::stream& strm) {
    const auto* probabilities = getSrcDataAtPortAs<const float>(DATA_INDEX);
    const auto* sequenceLengths = getSrcDataAtPortAs<const int>(SEQUENCE_LENGTH_INDEX);
    auto* decodedClasses = getDstDataAtPortAs<int>(DECODED_CLASSES_INDEX);
    auto* decodedClassesLength = getDstDataAtPortAs<int>(DECODED_CLASSES_LENGTH_INDEX);

    const size_t B = getParentEdgeAt(DATA_INDEX)->getMemory().getStaticDims()[0];
    ;
    const size_t T = getParentEdgeAt(DATA_INDEX)->getMemory().getStaticDims()[1];
    ;
    const int C = getParentEdgeAt(DATA_INDEX)->getMemory().getStaticDims()[2];
    ;
    const size_t TC = T * C;

    int blankIndex = C - 1;
    if (inputShapes.size() > BLANK_INDEX) {
        blankIndex = (getSrcDataAtPortAs<const int>(BLANK_INDEX))[0];
    }

    size_t workAmount = 0;
    for (size_t b = 0; b < B; b++) {
        if (sequenceLengths[b] > static_cast<int>(T)) {
            std::string errorMsg =
                "Sequence length " + std::to_string(sequenceLengths[b]) +
                " cannot be greater than according decoded classes dimension size " +
                std::to_string(getChildEdgeAt(DECODED_CLASSES_INDEX)->getMemory().getStaticDims()[1]);
            THROW_CPU_NODE_ERR(errorMsg);
        }
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
            const float* probs = probabilities + b * TC + C * tStart;
            const size_t actualSeqLen = sequenceLengths[b];

            for (size_t t = tStart; t < actualSeqLen; ++t) {
                int maxClassIdx = 0;
                float maxProb = probs[0];
                probs++;

                for (int c = 1; c < C; c++, probs++) {
                    if (*probs > maxProb) {
                        maxClassIdx = c;
                        maxProb = *probs;
                    }
                }
                decodedClasses[outputIndex++] = maxClassIdx;

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
        const size_t actualSeqLen = sequenceLengths[b];
        int* shiftedOut = decodedClasses + b * T;

        for (size_t t = 0; t < actualSeqLen; ++t) {
            if (*shiftedOut != blankIndex && !(mergeRepeated && *shiftedOut == prevClassIdx)) {
                decodedClasses[outputIndex++] = *shiftedOut;
            }
            prevClassIdx = *shiftedOut;
            shiftedOut++;
        }
        std::fill(decodedClasses + outputIndex, decodedClasses + (b + 1) * T, -1);
        decodedClassesLength[b] = outputIndex - b * T;
    });
}

bool CTCGreedyDecoderSeqLen::created() const {
    return getType() == Type::CTCGreedyDecoderSeqLen;
}

void CTCGreedyDecoderSeqLen::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool CTCGreedyDecoderSeqLen::needPrepareParams() const {
    return false;
}

}  // namespace ov::intel_cpu::node
