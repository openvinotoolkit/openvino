// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "embedding_bag_sum.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class EmbeddingBagPackedSumImpl: public MKLDNNEmbeddingBagSum {
public:
    explicit EmbeddingBagPackedSumImpl(const CNNLayer* layer) :
            MKLDNNEmbeddingBagSum(layer, 2lu, 1lu, 2lu, 3lu) {
        auto indicesData = layer->insData[INDICES_IDX].lock();
        if (indicesData == nullptr)
            THROW_IE_EXCEPTION << "'" << layer->name << "' layer has nullable indices data.";
        if (indicesData->getTensorDesc().getDims().size() != 2)
            THROW_IE_EXCEPTION << "'" << layer->name << "' layer has indices data with invalid shape.";

        _indices = std::vector<std::vector<size_t>>(
            indicesData->getTensorDesc().getDims()[0],
            std::vector<size_t>(indicesData->getTensorDesc().getDims()[1], 0lu));
    }

    void initFromInputs(std::vector<Blob::Ptr>& inputs) override {
        // Initialize indices
        const size_t bagsNum = inputs[INDICES_IDX]->getTensorDesc().getDims()[0];
        const size_t batch = inputs[INDICES_IDX]->getTensorDesc().getDims()[1];
        if (inputs[INDICES_IDX]->getTensorDesc().getPrecision().size() == sizeof(INT32)) {
            const INT32* src = inputs[INDICES_IDX]->cbuffer().as<const INT32*>();
            for (size_t i = 0lu; i < bagsNum; i++) {
                size_t ibn = i * batch;
                for (size_t j = 0lu; j < batch; j++) {
                    _indices[i][j] = static_cast<size_t>(src[ibn + j]);
                }
            }
        } else if (inputs[INDICES_IDX]->getTensorDesc().getPrecision().size() == sizeof(UINT64)) {
            const UINT64* src = inputs[INDICES_IDX]->cbuffer().as<const UINT64*>();
            for (size_t i = 0lu; i < bagsNum; i++) {
                memcpy(_indices[i].data(), src + i * batch, batch * sizeof(UINT64));
            }
        }
    }

    void getIndices(size_t embIndex, const size_t*& indices, size_t& size, size_t& weightsIdx, bool& withWeights) override {
        if (embIndex >= _indices.size())
            THROW_IE_EXCEPTION << "Invalid embedding bag index.";

        withWeights = true;

        indices = _indices[embIndex].data();
        size = _indices[0].size();

        weightsIdx = embIndex * _indices[0].size();
    }

protected:
    std::vector<std::vector<size_t>> _indices;
};

REG_FACTORY_FOR(EmbeddingBagPackedSumImpl, EmbeddingBagPackedSum);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
