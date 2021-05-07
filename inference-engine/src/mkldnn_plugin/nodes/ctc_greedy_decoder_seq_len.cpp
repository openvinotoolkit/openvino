// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"
#include "ie_parallel.hpp"
#include <ngraph/op/ctc_greedy_decoder_seq_len.hpp>
#include <nodes/common/tensor_desc_creator.h>

#include <string>
#include <vector>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

using MKLDNNPlugin::TensorDescCreatorTypes;

class CTCGreedyDecoderSeqLenImpl: public ExtLayerBase {
public:
    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
        try {
            auto greedyDecOp = ngraph::as_type_ptr<const ngraph::op::v6::CTCGreedyDecoderSeqLen>(op);
            if (!greedyDecOp) {
                errorMessage = "Node is not an instance of the CTCGreedyDecoderSeqLen operation from operation set v6.";
                return false;
            }
        } catch (...) {
            return false;
        }

        return true;
    }

    explicit CTCGreedyDecoderSeqLenImpl(const std::shared_ptr<ngraph::Node>& op) : mergeRepeated_(true) {
        try {
            std::string errorMessage;
            if (!isSupportedOperation(op, errorMessage)) {
                IE_THROW(NotImplemented) << errorMessage;
            }

            std::string errPrefix = "CTCGreedyDecoderSeqLen layer with name '" + op->get_friendly_name() + "' ";
            if (op->get_input_size() < 2 || op->get_input_size() > 3)
                IE_THROW() << errPrefix << "has invalid number of input edges: " << op->get_input_size();
            if (op->get_output_size() != 2)
                IE_THROW() << errPrefix << "has invalid number of outputs edges: " << op->get_output_size();

            if (op->get_input_shape(DATA_INDEX)[0] != op->get_input_shape(SEQUENCE_LENGTH_INDEX)[0])
                IE_THROW() << errPrefix << "has invalid input shapes.";

            Precision inDataPrecision = details::convertPrecision(op->get_input_element_type(DATA_INDEX));
            if (inDataPrecision != Precision::FP32 && inDataPrecision != Precision::BF16)
                IE_THROW() << errPrefix << "has unsupported 'data' input precision: " << inDataPrecision;

            Precision seqLenPrecision = details::convertPrecision(op->get_input_element_type(SEQUENCE_LENGTH_INDEX));
            if (seqLenPrecision != Precision::I32 && seqLenPrecision != Precision::I64)
                IE_THROW() << errPrefix << "has unsupported 'sequence_length' input precision: " << seqLenPrecision;

            auto greedyDecOp = ngraph::as_type_ptr<const ngraph::op::v6::CTCGreedyDecoderSeqLen>(op);
            mergeRepeated_ = greedyDecOp->get_merge_repeated();

            if (op->get_input_size() == BLANK_INDEX) {
                addConfig(op, {{TensorDescCreatorTypes::ncsp, Precision::FP32},
                               {TensorDescCreatorTypes::ncsp, Precision::I32}},
                              {{TensorDescCreatorTypes::ncsp, Precision::I32},
                               {TensorDescCreatorTypes::ncsp, Precision::I32}});
            } else {
                Precision blIdxPrecision = details::convertPrecision(op->get_input_element_type(BLANK_INDEX));
                if (blIdxPrecision != Precision::I32 && blIdxPrecision != Precision::I64)
                    IE_THROW() << errPrefix << "has unsupported 'blank_index' input precision: " << blIdxPrecision;

                addConfig(op, {{TensorDescCreatorTypes::ncsp, Precision::FP32},
                               {TensorDescCreatorTypes::ncsp, Precision::I32},
                               {TensorDescCreatorTypes::ncsp, Precision::I32}},
                              {{TensorDescCreatorTypes::ncsp, Precision::I32},
                               {TensorDescCreatorTypes::ncsp, Precision::I32}});
            }
        } catch (InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
            throw;
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        const float* probabilities = inputs[DATA_INDEX]->cbuffer().as<const float*>() +
            inputs[DATA_INDEX]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const int* sequenceLengths = inputs[SEQUENCE_LENGTH_INDEX]->cbuffer().as<const int*>() +
            inputs[SEQUENCE_LENGTH_INDEX]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        int* decodedClasses = outputs[DECODED_CLASSES_INDEX]->buffer().as<int*>() +
            outputs[DECODED_CLASSES_INDEX]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        int* decodedClassesLength = outputs[DECODED_CLASSES_LENGTH_INDEX]->buffer().as<int*>() +
            outputs[DECODED_CLASSES_LENGTH_INDEX]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        const auto& inDims = inputs[DATA_INDEX]->getTensorDesc().getDims();
        const size_t B = inDims[0];
        const size_t T = inDims[1];
        const int C = inDims[2];
        const size_t TC = T * C;

        int blankIndex = C - 1;
        if (inputs.size() > BLANK_INDEX)
            blankIndex = (inputs[BLANK_INDEX]->cbuffer().as<const int*>() +
                inputs[BLANK_INDEX]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0];

        size_t workAmount = 0;
        for (size_t b = 0; b < B; b++) {
            if (sequenceLengths[b] > T) {
                if (resp) {
                    std::string errorMsg = errPrefix
                        + ". Sequence length " + std::to_string(sequenceLengths[b])
                        + " cannot be greater than according decoded classes dimension size "
                        + std::to_string(outputs[DECODED_CLASSES_INDEX]->getTensorDesc().getDims()[1]);
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
                return PARAMETER_MISMATCH;
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
        }; // thread body

        parallel_nt(0, threadBody);

        parallel_for(B, [&](size_t b) {
            int prevClassIdx = -1;
            size_t outputIndex = b * T;
            const size_t actualSeqLen = sequenceLengths[b];
            int* shiftedOut = decodedClasses + b * T;

            for (size_t t = 0; t < actualSeqLen; ++t) {
                if (*shiftedOut != blankIndex &&
                        !(mergeRepeated_ && *shiftedOut == prevClassIdx)) {
                    decodedClasses[outputIndex++] = *shiftedOut;
                }
                prevClassIdx = *shiftedOut;
                shiftedOut++;
            }
            std::fill(decodedClasses + outputIndex, decodedClasses + (b + 1) * T, -1);
            decodedClassesLength[b] = outputIndex - b * T;
        });

        return OK;
    }

private:
    const size_t DATA_INDEX = 0lu;
    const size_t SEQUENCE_LENGTH_INDEX = 1lu;
    const size_t BLANK_INDEX = 2lu;
    const size_t DECODED_CLASSES_INDEX = 0lu;
    const size_t DECODED_CLASSES_LENGTH_INDEX = 1lu;
    bool mergeRepeated_;
    std::string errPrefix;
};

REG_FACTORY_FOR(CTCGreedyDecoderSeqLenImpl, CTCGreedyDecoderSeqLen);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
