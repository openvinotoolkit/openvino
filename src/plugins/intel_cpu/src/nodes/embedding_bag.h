// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class EmbeddingBag {
public:
    enum class Reduction { SUM, MEAN };
    EmbeddingBag(const std::shared_ptr<ov::Node>&,
                 size_t requiredInputsNum,
                 size_t indicesIdx,
                 size_t perSampleWeightsIdx,
                 size_t defaultIndexIdx);

    void execute(const uint8_t* srcData,
                 const uint8_t* weightsData,
                 const ov::element::Type& srcPrc,
                 const VectorDims& inDims,
                 const MemoryPtr& outMemory);

    ~EmbeddingBag() = default;

protected:
    virtual void initFromInputs() = 0;
    virtual void getIndices(size_t embIndex,
                            const int*& indicesRef,
                            size_t& size,
                            int& weightsIdx,
                            bool& withWeights) = 0;

    void prepareParams(const VectorDims& indexStaticShape);

    template <typename T>
    void processData(const T* srcData, const T* weightsData, const VectorDims& inDataDims, const MemoryPtr& outMemory);

    const size_t EMB_TABLE_IDX = 0lu;
    const size_t INDICES_IDX;
    const size_t PER_SAMPLE_WEIGHTS_IDX;
    const size_t DEFAULT_INDEX_IDX;

    Reduction _reduction = Reduction::SUM;
    bool _withWeights = false;
    size_t _embDepth = 0;
    std::string _layerName;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
