// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ctc_greedy_decoder_seq_len.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <openvino/op/ctc_greedy_decoder_seq_len.hpp>
#include <string>
#include <vector>

#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/general_utils.h"

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

    CPU_NODE_ASSERT(getOriginalInputsNumber() >= 2 && getOriginalInputsNumber() <= 3,
                    "has invalid number of input edges: ",
                    getOriginalInputsNumber());
    CPU_NODE_ASSERT(getOriginalOutputsNumber() == 2,
                    "has invalid number of outputs edges: ",
                    getOriginalOutputsNumber());

    const auto& dataDims = getInputShapeAtPort(DATA_INDEX).getDims();
    const auto& seqDims = getInputShapeAtPort(SEQUENCE_LENGTH_INDEX).getDims();
    CPU_NODE_ASSERT(dimsEqualWeak(dataDims[0], seqDims[0]), "has invalid input shapes.");

    auto greedyDecOp = ov::as_type_ptr<const ov::op::v6::CTCGreedyDecoderSeqLen>(op);
    mergeRepeated = greedyDecOp->get_merge_repeated();
}

void CTCGreedyDecoderSeqLen::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    ov::element::Type inDataPrecision = getOriginalInputPrecisionAtPort(DATA_INDEX);
    CPU_NODE_ASSERT(any_of(inDataPrecision, ov::element::f32, ov::element::bf16, ov::element::f16),
                    "has unsupported 'data' input precision: ",
                    inDataPrecision);

    ov::element::Type seqLenPrecision = getOriginalInputPrecisionAtPort(SEQUENCE_LENGTH_INDEX);
    CPU_NODE_ASSERT(any_of(seqLenPrecision, ov::element::i32, ov::element::i64),
                    "has unsupported 'sequence_length' input precision: ",
                    seqLenPrecision);

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

void CTCGreedyDecoderSeqLen::execute([[maybe_unused]] const dnnl::stream& strm) {
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
        CPU_NODE_ASSERT(sequenceLengths[b] <= static_cast<int>(T),
                        "Sequence length ",
                        sequenceLengths[b],
                        " cannot be greater than according decoded classes dimension size ",
                        getChildEdgeAt(DECODED_CLASSES_INDEX)->getMemory().getStaticDims()[1]);
        workAmount += sequenceLengths[b];
    }
    // Parallelization could not be made directly by T due to output index depends on merged classes and
    // blank index, thus could not be shared between threads. Better to divide operation on two steps.
    // At the first stage find the maximum index. At second stage merge if needed.
    // Such approach makes parallelization more efficient.
    auto threadBody = [&](const int ithr, const int nthr) {
        size_t start(0LU);
        size_t end(0LU);
        splitter(workAmount, nthr, ithr, start, end);
        if (start >= end) {
            return;
        }
        size_t tStart = 0LU;
        size_t bStart = 0LU;
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
            tStart = 0LU;
        }
    };  // thread body

    parallel_nt(0, threadBody);

    parallel_nt(B, [&](size_t b, int nthr) {
        int prevClassIdx = -1;
        size_t outputIndex = b * T;
        const size_t actualSeqLen = sequenceLengths[b];
        int* shiftedOut = decodedClasses + b * T;

        for (size_t t = 0; t < actualSeqLen; ++t) {
            if (*shiftedOut != blankIndex && (!mergeRepeated || *shiftedOut != prevClassIdx)) {
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
