// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "embedding_bag_sum.hpp"
#include "ie_parallel.hpp"

#include <vector>


namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class EmbeddingBagOffsetsSumImpl: public MKLDNNEmbeddingBagSum {
public:
    explicit EmbeddingBagOffsetsSumImpl(const CNNLayer* layer) :
                MKLDNNEmbeddingBagSum(layer, 3lu, 1lu, 4lu, 3lu) {
        auto indicesData = layer->insData[INDICES_IDX].lock();
        if (indicesData == nullptr)
            THROW_IE_EXCEPTION << "'" << layer->name << "' layer has nullable indices data.";
        if (indicesData->getTensorDesc().getDims().size() != 1)
            THROW_IE_EXCEPTION << "'" << layer->name << "' layer has indices data with invalid shape.";

        auto offsetsData = layer->insData[OFFSETS_IDX].lock();
        if (offsetsData == nullptr)
            THROW_IE_EXCEPTION << "'" << layer->name << "' layer has invalid offsets data.";
        if (offsetsData->getTensorDesc().getDims().size() != 1)
            THROW_IE_EXCEPTION << "'" << layer->name << "' layer's offsets data has invalid shape.";

        _indicesLen = indicesData->getTensorDesc().getDims()[0];
        _offsetsLen = offsetsData->getTensorDesc().getDims()[0];
    }

    StatusCode execute(
                std::vector<Blob::Ptr>& inputs,
                std::vector<Blob::Ptr>& outputs,
                ResponseDesc* resp) noexcept override {
        switch (inputs[0]->getTensorDesc().getPrecision()) {
            case Precision::FP32: {
                return processData<PrecisionTrait<Precision::FP32>::value_type>(inputs, outputs, resp);
            }
            case Precision::I8: {
                return processData<PrecisionTrait<Precision::I8>::value_type>(inputs, outputs, resp);
            }
            case Precision::U8: {
                return processData<PrecisionTrait<Precision::U8>::value_type>(inputs, outputs, resp);
            }
            case Precision::I32: {
                return processData<PrecisionTrait<Precision::I32>::value_type>(inputs, outputs, resp);
            }
            default: {
                if (resp) {
                    std::string errorMsg = "EmbeddingBagSum layer does not support embedding table precision '"
                            + std::string(inputs[0]->getTensorDesc().getPrecision().name()) + "'";
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
                return GENERAL_ERROR;
            }
        }
    }

protected:
    template<typename T>
    StatusCode processData(
                std::vector<Blob::Ptr>& inputs,
                std::vector<Blob::Ptr>& outputs,
                ResponseDesc* resp) noexcept {
        switch (inputs[1]->getTensorDesc().getPrecision()) {
            case Precision::I32: {
                return processData<T, PrecisionTrait<Precision::I32>::value_type>(inputs, outputs, resp);
            }
            case Precision::I64: {
                return processData<T, PrecisionTrait<Precision::I64>::value_type>(inputs, outputs, resp);
            }
            case Precision::U64: {
                return processData<T, PrecisionTrait<Precision::U64>::value_type>(inputs, outputs, resp);
            }
            default: {
                if (resp) {
                    std::string errorMsg = "EmbeddingBagSum layer does not support indices precision '"
                            + std::string(inputs[1]->getTensorDesc().getPrecision().name()) + "'";
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
                return GENERAL_ERROR;
            }
        }
    }

    template<typename T, typename I>
    StatusCode processData(
                std::vector<Blob::Ptr>& inputs,
                std::vector<Blob::Ptr>& outputs,
                ResponseDesc* resp) noexcept {
        std::string errorMsg;
        std::string msgPrefix = std::string("Layer EmbeddingBagOffsetsSum with name '") + _layerName + "' ";

        const T* srcData = inputs[0]->cbuffer().as<const T*>() +
            inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        T* dstData = outputs[0]->buffer().as<T*>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        const I* indicesData = inputs[INDICES_IDX]->cbuffer().as<const I*>();

        const I* offsetsData = inputs[OFFSETS_IDX]->cbuffer().as<const I*>();
        int64_t defaultIndex = -1;
        if (inputs.size() > DEFAULT_INDEX_IDX) {
            defaultIndex = (int64_t)inputs[DEFAULT_INDEX_IDX]->cbuffer().as<const I*>()[0];
            if (defaultIndex < 0 || defaultIndex >= _indicesLen) {
                std::string msg =  "Invalid default index: " + std::to_string(defaultIndex);
                msg.copy(resp->msg, sizeof(resp->msg) - 1);
                return GENERAL_ERROR;
            }
        }
        const T* weightsData = nullptr;
        if (_withWeights)
            weightsData = inputs[PER_SAMPLE_WEIGHTS_IDX]->cbuffer().as<const T*>();

        const auto& inDataDims = inputs[0]->getTensorDesc().getDims();

        const size_t OUTPUT_BAGS_NUM = outputs[0]->getTensorDesc().getDims()[0];

        std::function<void(size_t, const I*&, size_t&, size_t&, bool&)> get_idx =
                [&](size_t embIndex, const I*& indicesRef, size_t& outSize, size_t& weightsIdx, bool& withWeights) {
            if (embIndex >= _offsetsLen) {
                errorMsg = msgPrefix + "has invalid embedding bag index.";
                return;
            }
            if (offsetsData[embIndex] >= _indicesLen) {
                errorMsg = msgPrefix + ". Offset value exceeds indices size in the model.\noffset: "
                    + std::to_string(offsetsData[embIndex]) + "; indices size: " + std::to_string(_indicesLen);
                return;
            }

            indicesRef = nullptr;
            outSize = 0lu;
            withWeights = _withWeights;

            if (embIndex == _offsetsLen - 1lu)
                outSize = _indicesLen - offsetsData[embIndex];
            else
                outSize = offsetsData[embIndex + 1lu] - offsetsData[embIndex];

            if (outSize != 0lu) {
                indicesRef = indicesData + offsetsData[embIndex];
            } else {
            // Empty or default bag
                withWeights = false;
                if (defaultIndex >= 0) {
                    indicesRef = reinterpret_cast<I*>(&defaultIndex);
                    outSize = 1lu;
                }
                return;
            }

            if (withWeights)
                weightsIdx = offsetsData[embIndex];
        };

        auto threadBody = [&](const int ithr, const int nthr) {
            size_t start(0lu), end(0lu);
            splitter(OUTPUT_BAGS_NUM, nthr, ithr, start, end);
            if (start >= end)
                return;

            size_t indicesSize = 0lu;
            const I* indices = nullptr;
            size_t weightsIdx = 0lu;
            bool withWeights = _withWeights;

            for (size_t obi = start; obi < end; obi++) {
                size_t dstIndex = obi * _embDepth;
                get_idx(obi, indices, indicesSize, weightsIdx, withWeights);
                if (indices != nullptr) {
                    withWeights = withWeights & _withWeights;

                    size_t inIdx = 0lu;
                    if (indices[inIdx] >= inDataDims[0]) {
                        errorMsg = msgPrefix + "has invalid embedding bag index: " + std::to_string(indices[inIdx]);
                        return;
                    }
                    size_t srcIndex = indices[inIdx] * _embDepth;

                    if (withWeights) {
                        for (size_t i = 0lu; i < _embDepth; i++) {
                            dstData[dstIndex + i] = srcData[srcIndex + i] * weightsData[weightsIdx];
                        }
                        weightsIdx++;
                    } else {
                        for (size_t i = 0lu; i < _embDepth; i++) {
                            dstData[dstIndex + i] = srcData[srcIndex + i];
                        }
                    }

                    for (inIdx = 1lu; inIdx < indicesSize; inIdx++) {
                        if (indices[inIdx] >= inDataDims[0]) {
                            errorMsg = msgPrefix + "has invalid embedding bag index: " + std::to_string(indices[inIdx]);
                            return;
                        }
                        size_t srcIndex = indices[inIdx] * _embDepth;

                        if (withWeights) {
                            for (size_t i = 0lu; i < _embDepth; i++) {
                                dstData[dstIndex + i] += srcData[srcIndex + i] * weightsData[weightsIdx];
                            }
                            weightsIdx++;
                        } else {
                            for (size_t i = 0lu; i < _embDepth; i++) {
                                dstData[dstIndex + i] += srcData[srcIndex + i];
                            }
                        }
                    }
                } else {
                    for (size_t i = 0lu; i < _embDepth; i++) {
                        dstData[dstIndex + i] = 0;
                    }
                }
            }
        };

        parallel_nt(0, threadBody);

        if (!errorMsg.empty()) {
            errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            return GENERAL_ERROR;
        }

        return OK;
    }

    void initFromInputs(std::vector<Blob::Ptr>& inputs) override {
    }

    void getIndices(size_t embIndex, const size_t*& indices, size_t& size, size_t& weightsIdx, bool& withWeights) override {
    }

    const size_t OFFSETS_IDX = 2lu;

    size_t _indicesLen;
    size_t _offsetsLen;
};

REG_FACTORY_FOR(EmbeddingBagOffsetsSumImpl, EmbeddingBagOffsetsSum);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
