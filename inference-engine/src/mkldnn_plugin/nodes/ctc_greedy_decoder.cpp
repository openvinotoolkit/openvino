// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"
#include "ie_parallel.hpp"
#include <ngraph/op/ctc_greedy_decoder.hpp>
#include <nodes/common/tensor_desc_creator.h>

#include <string>
#include <vector>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

using MKLDNNPlugin::TensorDescCreatorTypes;

class CTCGreedyDecoderImpl: public ExtLayerBase {
public:
    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
        try {
            auto greedyDecOp = ngraph::as_type_ptr<const ngraph::op::v0::CTCGreedyDecoder>(op);
            if (!greedyDecOp) {
                errorMessage = "Node is not an instance of the CTCGreedyDecoder operation from operation set v0.";
                return false;
            }
        } catch (...) {
            return false;
        }

        return true;
    }

    explicit CTCGreedyDecoderImpl(const std::shared_ptr<ngraph::Node>& op) : mergeRepeated_(true) {
        try {
            std::string errorMessage;
            if (!isSupportedOperation(op, errorMessage)) {
                IE_THROW(NotImplemented) << errorMessage;
            }

            std::string errPrefix = "CTCGreedyDecoder layer with name '" + op->get_friendly_name() + "' ";
            if (op->get_input_size() != 2)
                IE_THROW() << errPrefix << "has invalid number of input edges: " << op->get_input_size();
            if (op->get_output_size() != 1)
                IE_THROW() << errPrefix << "has invalid number of outputs edges: " << op->get_output_size();

            if (op->get_input_shape(DATA_INDEX)[0] != op->get_input_shape(SEQUENCE_LENGTH_INDEX)[0] &&
                    op->get_input_shape(DATA_INDEX)[1] != op->get_input_shape(SEQUENCE_LENGTH_INDEX)[1])
                IE_THROW() << errPrefix << "has invalid input shapes.";

            Precision inDataPrecision = details::convertPrecision(op->get_input_element_type(DATA_INDEX));
            if (inDataPrecision != Precision::FP32 && inDataPrecision != Precision::BF16)
                IE_THROW() << errPrefix << "has unsupported 'data' input precision: " << inDataPrecision;

            Precision seqLenPrecision = details::convertPrecision(op->get_input_element_type(SEQUENCE_LENGTH_INDEX));
            if (seqLenPrecision != Precision::FP32 && seqLenPrecision != Precision::BF16)
                IE_THROW() << errPrefix << "has unsupported 'sequence_length' input precision: " << seqLenPrecision;

            auto greedyDecOp = ngraph::as_type_ptr<const ngraph::op::v0::CTCGreedyDecoder>(op);
            mergeRepeated_ = greedyDecOp->get_ctc_merge_repeated();

            addConfig(op, {{TensorDescCreatorTypes::ncsp, Precision::FP32},
                           {TensorDescCreatorTypes::ncsp, Precision::FP32}},
                          {{TensorDescCreatorTypes::ncsp, Precision::FP32}});
        } catch (InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
            throw;
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        const float* probabilities = inputs[DATA_INDEX]->cbuffer().as<const float*>() +
            inputs[DATA_INDEX]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const float* sequenceMask = inputs[SEQUENCE_LENGTH_INDEX]->cbuffer().as<const float*>() +
            inputs[SEQUENCE_LENGTH_INDEX]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        float* outputSequences = outputs[0]->buffer().as<float*>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        const size_t T = inputs[DATA_INDEX]->getTensorDesc().getDims()[0];
        const size_t B = inputs[DATA_INDEX]->getTensorDesc().getDims()[1];
        const int C = inputs[DATA_INDEX]->getTensorDesc().getDims()[2];
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
                        !(mergeRepeated_ && *shiftedOut == prevClassIdx)) {
                    outputSequences[outputIndex++] = *shiftedOut;
                }
                prevClassIdx = *shiftedOut;
                shiftedOut++;
            }
            std::fill(outputSequences + outputIndex, outputSequences + (b + 1) * T, -1.f);
        });

        return OK;
    }

private:
    const size_t DATA_INDEX = 0lu;
    const size_t SEQUENCE_LENGTH_INDEX = 1lu;
    bool mergeRepeated_;
};

REG_FACTORY_FOR(CTCGreedyDecoderImpl, CTCGreedyDecoder);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
