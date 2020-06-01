// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base.hpp"

#include <memory>
#include <set>
#include <vector>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class MKLDNNEmbeddingBagSum : public ExtLayerBase {
public:
    MKLDNNEmbeddingBagSum(
        const CNNLayer* layer,
        size_t requiredInputsNum,
        size_t indicesIdx,
        size_t perSampleWeightsIdx,
        size_t defaultIndexIdx,
        const std::set<Precision>& supportedPrecisions = {});

    StatusCode execute(
        std::vector<Blob::Ptr>& inputs,
        std::vector<Blob::Ptr>& outputs,
        ResponseDesc *resp) noexcept override;

protected:
    virtual void initFromInputs(std::vector<Blob::Ptr>& inputs) = 0;
    virtual void getIndices(
        size_t embIndex,
        const size_t*& indicesRef,
        size_t& size,
        size_t& weightsIdx,
        bool& withWeights) = 0;

    template<typename T>
    void processData(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs) noexcept;

    std::set<Precision> _supportedPrecisions;

    const size_t INDICES_IDX;
    const size_t PER_SAMPLE_WEIGHTS_IDX;
    const size_t DEFAULT_INDEX_IDX;

    bool _withWeights = false;
    size_t _embDepth = 0;
    std::string _layerName;

    using INT32 = PrecisionTrait<Precision::I32>::value_type;
    using INT64 = PrecisionTrait<Precision::I64>::value_type;
    using UINT64 = PrecisionTrait<Precision::U64>::value_type;

    static const std::set<size_t> _supportedIndicesTypeSize;
};

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
