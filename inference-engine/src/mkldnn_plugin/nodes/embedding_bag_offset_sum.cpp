// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "embedding_bag_sum.hpp"
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

        _indices = std::vector<size_t>(indicesData->getTensorDesc().getDims()[0], 0lu);
        _offsets = std::vector<size_t>(offsetsData->getTensorDesc().getDims()[0], 0lu);
    }

protected:
    void init_from_inputs(std::vector<Blob::Ptr>& inputs) override {
        // Initialize indices
        if (inputs[INDICES_IDX]->getTensorDesc().getPrecision().size() == sizeof(INT32)) {
            const INT32* src = inputs[INDICES_IDX]->cbuffer().as<const INT32*>();
            for (size_t i = 0lu; i < inputs[INDICES_IDX]->size(); i++)
                _indices[i] = static_cast<size_t>(src[i]);
        } else if (inputs[INDICES_IDX]->getTensorDesc().getPrecision().size() == sizeof(UINT64)) {
            const UINT64* src = inputs[INDICES_IDX]->cbuffer().as<const UINT64*>();
            memcpy(_indices.data(), src, inputs[INDICES_IDX]->byteSize());
        }

        // Initialize offsets
        if (inputs[OFFSETS_IDX]->getTensorDesc().getPrecision().size() == sizeof(INT32)) {
            const INT32* src = inputs[OFFSETS_IDX]->cbuffer().as<const INT32*>();
            for (size_t i = 0lu; i < inputs[OFFSETS_IDX]->size(); i++)
                _offsets[i] = static_cast<size_t>(src[i]);
        } else if (inputs[OFFSETS_IDX]->getTensorDesc().getPrecision().size() == sizeof(UINT64)) {
            const UINT64* src = inputs[OFFSETS_IDX]->cbuffer().as<const UINT64*>();
            memcpy(_offsets.data(), src, inputs[OFFSETS_IDX]->byteSize());
        }

        // Initialize default index
        _default_indices.clear();
        if (inputs.size() > DEFAULT_INDEX_IDX) {
            if (inputs[DEFAULT_INDEX_IDX]->getTensorDesc().getPrecision().size() == sizeof(INT32)) {
                const INT32* src = inputs[DEFAULT_INDEX_IDX]->cbuffer().as<const INT32*>();
                _default_indices.push_back(static_cast<size_t>(*src));
            } else if (inputs[DEFAULT_INDEX_IDX]->getTensorDesc().getPrecision().size() == sizeof(UINT64)) {
                const INT64* src = inputs[DEFAULT_INDEX_IDX]->cbuffer().as<const INT64*>();
                _default_indices.push_back(*src);
            }
        }
    }

    void get_indices(size_t emb_index, const size_t*& indices, size_t& size, size_t& weights_idx, bool& with_weights) override {
        if (emb_index >= _offsets.size())
            THROW_IE_EXCEPTION << "Layer EmbeddingBagOffsetsSum with name '" << _l_name
                << "' has invalid embedding bag index.";
        if (_offsets[emb_index] >= _indices.size())
            THROW_IE_EXCEPTION << "Layer EmbeddingBagOffsetsSum with name '" << _l_name
                << "'. Offset value exceeds indices size in the model.\noffset: "
                << _offsets[emb_index] << "; indices size: " << _indices.size();

        indices = nullptr;
        size = 0lu;
        with_weights = _with_weights;

        if (emb_index == _offsets.size() - 1lu)
            size = _indices.size() - _offsets[emb_index];
        else
            size = _offsets[emb_index + 1lu] - _offsets[emb_index];

        if (size != 0lu) {
            indices = _indices.data() + _offsets[emb_index];
        } else {
        // Empty or default bag
            with_weights = false;
            if (_default_indices.size() == 1lu) {
                indices = _default_indices.data();
                size = 1lu;
            }
            return;
        }

        if (with_weights)
            weights_idx = _offsets[emb_index];
    }

protected:
    const size_t OFFSETS_IDX = 2lu;

    std::vector<size_t> _indices;
    std::vector<size_t> _offsets;
    std::vector<size_t> _default_indices;
};

REG_FACTORY_FOR(EmbeddingBagOffsetsSumImpl, EmbeddingBagOffsetsSum);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine

