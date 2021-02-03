// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"
#include "ie_parallel.hpp"

#include <vector>
#include <string>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class CTCGreedyDecoderSeqLenImpl: public ExtLayerBase {
public:
    explicit CTCGreedyDecoderSeqLenImpl(const CNNLayer* layer) : mergeRepeated_(true) {
        std::string errPrefix = "CTCGreedyDecoderSeqLen layer with name '" + layer->name + "' ";
        if (layer->insData.size() < 2 || layer->insData.size() > 3)
            THROW_IE_EXCEPTION << errPrefix << "has invalid number of input edges: " << layer->insData.size();
        if (layer->outData.size() != 2)
            THROW_IE_EXCEPTION << errPrefix << "has invalid number of outputs edges: " << layer->outData.size();

        auto inData = layer->insData[DATA_INDEX].lock();
        auto sequenceLenData = layer->insData[SEQUENCE_LENGTH_INDEX].lock();
        if (!inData || !sequenceLenData)
            THROW_IE_EXCEPTION << errPrefix << "has nullable inputs.";
        if (inData->getTensorDesc().getDims()[0] != sequenceLenData->getTensorDesc().getDims()[0])
            THROW_IE_EXCEPTION << errPrefix << "has invalid input shapes.";
        if (inData->getTensorDesc().getPrecision() != Precision::FP32 &&
                inData->getTensorDesc().getPrecision() != Precision::BF16)
            THROW_IE_EXCEPTION << errPrefix << "has unsupported 'data' input precision: " << inData->getTensorDesc().getPrecision();
        if (sequenceLenData->getTensorDesc().getPrecision() != Precision::I32 &&
                sequenceLenData->getTensorDesc().getPrecision() != Precision::I64)
            THROW_IE_EXCEPTION << errPrefix << "has unsupported 'sequence_length' input precision: " << sequenceLenData->getTensorDesc().getPrecision();

        std::vector<DataConfigurator> inputConfigs{{ConfLayout::PLN, Precision::FP32}, {ConfLayout::PLN, Precision::I32}};

        if (layer->insData.size() > BLANK_INDEX) {
            auto blankIndexData = layer->insData[BLANK_INDEX].lock();
            if (!blankIndexData)
                THROW_IE_EXCEPTION << errPrefix << "has nullable inputs.";
            if (blankIndexData->getTensorDesc().getPrecision() != Precision::I32 &&
                    blankIndexData->getTensorDesc().getPrecision() != Precision::I64)
                THROW_IE_EXCEPTION << errPrefix << "has unsupported 'blank_index' input precision: " << blankIndexData->getTensorDesc().getPrecision();
            inputConfigs.push_back({ConfLayout::PLN, Precision::I32});
        }
        std::vector<DataConfigurator> outputConfigs{{ConfLayout::PLN, Precision::I32}, {ConfLayout::PLN, Precision::I32}};
        addConfig(layer, inputConfigs, outputConfigs);

        mergeRepeated_ = layer->GetParamAsBool("merge_repeated", true);
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
};

REG_FACTORY_FOR(CTCGreedyDecoderSeqLenImpl, CTCGreedyDecoderSeqLen);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
