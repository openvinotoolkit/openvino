// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "embedding_bag_sum.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class EmbeddingSegmentsSumImpl: public MKLDNNEmbeddingBagSum {
public:
    explicit EmbeddingSegmentsSumImpl(const CNNLayer* layer) :
                MKLDNNEmbeddingBagSum(layer, 4lu, 1lu, 5lu, 4lu) {
        std::string errPrefix = std::string("EmbeddingSegmentsSum layer with name '") + _layerName + "' ";
        auto indicesData = layer->insData[INDICES_IDX].lock();
        if (indicesData == nullptr)
            THROW_IE_EXCEPTION << errPrefix << "has nullable indices data.";
        if (indicesData->getTensorDesc().getDims().size() != 1)
            THROW_IE_EXCEPTION << errPrefix << "has indices data with invalid shape: "
                << indicesData->getTensorDesc().getDims().size();

        auto segmentIdData = layer->insData[SEGMENT_ID_IDX].lock();
        if (segmentIdData == nullptr)
            THROW_IE_EXCEPTION << errPrefix << "has invalid segmentID data.";
        if (segmentIdData->getTensorDesc().getDims().size() != 1)
            THROW_IE_EXCEPTION << errPrefix << "has invalid segmentID data shape: "
                << segmentIdData->getTensorDesc().getDims().size();

        auto numSegmentData = layer->insData[NUM_SEGMENTS_IDX].lock();
        if (numSegmentData == nullptr)
            THROW_IE_EXCEPTION << errPrefix << "has nullable numSegmentID data.";

        if (_supportedIndicesTypeSize.find(indicesData->getTensorDesc().getPrecision().size())
                    == _supportedIndicesTypeSize.end()
                || _supportedIndicesTypeSize.find(segmentIdData->getTensorDesc().getPrecision().size())
                    == _supportedIndicesTypeSize.end()
                || _supportedIndicesTypeSize.find(numSegmentData->getTensorDesc().getPrecision().size())
                    == _supportedIndicesTypeSize.end())
            THROW_IE_EXCEPTION << errPrefix << "has unsupported input data type.";

        _indices = std::vector<size_t>(indicesData->getTensorDesc().getDims()[0], 0lu);
        _segmentIds = std::vector<size_t>(segmentIdData->getTensorDesc().getDims()[0], 0lu);
    }

    void initFromInputs(std::vector<Blob::Ptr>& inputs) override {
        // Initialize indices
        if (inputs[INDICES_IDX]->getTensorDesc().getPrecision().size() == sizeof(INT32)) {
            const INT32* src = inputs[INDICES_IDX]->cbuffer().as<const INT32*>();
            for (size_t i = 0lu; i < inputs[INDICES_IDX]->size(); i++)
                _indices[i] = static_cast<size_t>(src[i]);
        } else if (inputs[INDICES_IDX]->getTensorDesc().getPrecision().size() == sizeof(UINT64)) {
            const UINT64* src = inputs[INDICES_IDX]->cbuffer().as<const UINT64*>();
            memcpy(_indices.data(), src, inputs[INDICES_IDX]->byteSize());
        }

        // Initialize segments ids
        if (inputs[SEGMENT_ID_IDX]->getTensorDesc().getPrecision().size() == sizeof(INT32)) {
            const INT32* src = inputs[SEGMENT_ID_IDX]->cbuffer().as<const INT32*>();
            for (size_t i = 0lu; i < inputs[SEGMENT_ID_IDX]->size(); i++)
                _segmentIds[i] = static_cast<size_t>(src[i]);
        } else if (inputs[SEGMENT_ID_IDX]->getTensorDesc().getPrecision().size() == sizeof(UINT64)) {
            const UINT64* src = inputs[SEGMENT_ID_IDX]->cbuffer().as<const UINT64*>();
            memcpy(_segmentIds.data(), src, inputs[SEGMENT_ID_IDX]->byteSize());
        }

        if (inputs.size() > NUM_SEGMENTS_IDX) {
            if (inputs[NUM_SEGMENTS_IDX]->getTensorDesc().getPrecision().size() == sizeof(INT32)) {
                const INT32* src = inputs[NUM_SEGMENTS_IDX]->cbuffer().as<const INT32*>();
                _numSegments = static_cast<size_t>(*src);
            } else if (inputs[NUM_SEGMENTS_IDX]->getTensorDesc().getPrecision().size() == sizeof(UINT64)) {
                const INT64* src = inputs[NUM_SEGMENTS_IDX]->cbuffer().as<const INT64*>();
                _numSegments = *src;
            }
        }

        // Initialize default index
        _defaultIndices.clear();
        if (inputs.size() > DEFAULT_INDEX_IDX) {
            if (inputs[DEFAULT_INDEX_IDX]->getTensorDesc().getPrecision().size() == sizeof(INT32)) {
                const INT32* src = inputs[DEFAULT_INDEX_IDX]->cbuffer().as<const INT32*>();
                _defaultIndices.push_back(static_cast<size_t>(*src));
            } else if (inputs[DEFAULT_INDEX_IDX]->getTensorDesc().getPrecision().size() == sizeof(UINT64)) {
                const INT64* src = inputs[DEFAULT_INDEX_IDX]->cbuffer().as<const INT64*>();
                _defaultIndices.push_back(*src);
            }
        }
    }

    void getIndices(size_t embIndex, const size_t*& indices, size_t& size, size_t& weightsIdx, bool& withWeight) override {
        if (embIndex >= _numSegments)
            THROW_IE_EXCEPTION << "Invalid embedding bag index.";

        indices = nullptr;
        size = 0lu;
        withWeight = true;

        for (size_t si = 0; si < _indices.size(); si++) {
            if (_segmentIds[si] == embIndex) {
                size++;
                if (indices == nullptr) {
                    indices = _indices.data() + si;
                    weightsIdx = si;
                }
            }
        }

        // Empty bag
        if (size == 0) {
            size = 1lu;
            withWeight = false;
            if (_defaultIndices.size() == 1lu)
                indices = _defaultIndices.data();
            return;
        }
    }

protected:
    const size_t SEGMENT_ID_IDX = 2lu;
    const size_t NUM_SEGMENTS_IDX = 3lu;

    size_t _numSegments = 0lu;

    std::vector<size_t> _indices;
    std::vector<size_t> _segmentIds;
    std::vector<size_t> _defaultIndices;
};

REG_FACTORY_FOR(EmbeddingSegmentsSumImpl, EmbeddingSegmentsSum);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
