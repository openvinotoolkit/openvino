// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include <ngraph/op/ctc_greedy_decoder.hpp>
#include "ie_parallel.hpp"
#include "mkldnn_ctc_greedy_decoder_node.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNCTCGreedyDecoderNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto greedyDecOp = ngraph::as_type_ptr<const ngraph::op::v0::CTCGreedyDecoder>(op);
        if (!greedyDecOp) {
            errorMessage = "Node is not an instance of the CTCGreedyDecoder operation from operation set v0.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNCTCGreedyDecoderNode::MKLDNNCTCGreedyDecoderNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "CTCGreedyDecoder layer with name '" + op->get_friendly_name() + "' ";
    if (getOriginalInputsNumber() != 2)
        IE_THROW() << errorPrefix << "has invalid number of input edges: " << getOriginalInputsNumber();
    if (getOriginalOutputsNumber() != 1)
        IE_THROW() << errorPrefix << "has invalid number of outputs edges: " << getOriginalOutputsNumber();

    if (op->get_input_shape(DATA_INDEX)[0] != op->get_input_shape(SEQUENCE_LENGTH_INDEX)[0] &&
        op->get_input_shape(DATA_INDEX)[1] != op->get_input_shape(SEQUENCE_LENGTH_INDEX)[1])
        IE_THROW() << errorPrefix << "has invalid input shapes.";

    auto greedyDecOp = ngraph::as_type_ptr<const ngraph::op::v0::CTCGreedyDecoder>(op);
    mergeRepeated = greedyDecOp->get_ctc_merge_repeated();
}

void MKLDNNCTCGreedyDecoderNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    Precision inDataPrecision = getOriginalInputPrecisionAtPort(DATA_INDEX);
    if (inDataPrecision != Precision::FP32 && inDataPrecision != Precision::BF16)
        IE_THROW() << errorPrefix << "has unsupported 'data' input precision: " << inDataPrecision;

    Precision seqLenPrecision = getOriginalInputPrecisionAtPort(SEQUENCE_LENGTH_INDEX);
    if (seqLenPrecision != Precision::FP32 && seqLenPrecision != Precision::BF16)
        IE_THROW() << errorPrefix << "has unsupported 'sequence_length' input precision: " << seqLenPrecision;

    addSupportedPrimDesc({{GeneralLayout::ncsp, Precision::FP32},
                          {GeneralLayout::ncsp, Precision::FP32}},
                         {{GeneralLayout::ncsp, Precision::FP32}},
                         impl_desc_type::ref_any);
}

void MKLDNNCTCGreedyDecoderNode::execute(mkldnn::stream strm) {
    const float* probabilities = reinterpret_cast<const float *>(getParentEdgeAt(DATA_INDEX)->getMemoryPtr()->GetPtr());
    const float* sequenceMask = reinterpret_cast<const float *>(getParentEdgeAt(SEQUENCE_LENGTH_INDEX)->getMemoryPtr()->GetPtr());
    float* outputSequences = reinterpret_cast<float *>(getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPtr());

    const size_t T = getParentEdgeAt(DATA_INDEX)->getShape().getStaticDims()[0];
    const size_t B = getParentEdgeAt(DATA_INDEX)->getShape().getStaticDims()[1];
    const int C = getParentEdgeAt(DATA_INDEX)->getShape().getStaticDims()[2];
    const size_t BC = B * C;
    const size_t CB1 = C * (B - 1);

    const int blankIndex = C - 1;

    std::vector<size_t> sequenceLengths(B, 0);
    parallel_for(B, [&](size_t b) {
        size_t t = 0;
        for (; t < T; t++) {
            if (sequenceMask[B * t + b] == 0.f)
                break;
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
        if (start >= end)
            return;
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
    }; // thread body

    parallel_nt(0, threadBody);

    parallel_for(B, [&](size_t b) {
        int prevClassIdx = -1;
        size_t outputIndex = b * T;
        const size_t sequenceLength = sequenceLengths[b];
        float* shiftedOut = outputSequences + b * T;
        for (size_t t = 0; t < sequenceLength; ++t) {
            if (*shiftedOut < blankIndex &&
                !(mergeRepeated && *shiftedOut == prevClassIdx)) {
                outputSequences[outputIndex++] = *shiftedOut;
            }
            prevClassIdx = *shiftedOut;
            shiftedOut++;
        }
        std::fill(outputSequences + outputIndex, outputSequences + (b + 1) * T, -1.f);
    });
}

bool MKLDNNCTCGreedyDecoderNode::created() const {
    return getType() == CTCGreedyDecoder;
}

REG_MKLDNN_PRIM_FOR(MKLDNNCTCGreedyDecoderNode, CTCGreedyDecoder)
