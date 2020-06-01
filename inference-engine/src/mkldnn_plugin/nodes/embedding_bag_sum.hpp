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
        size_t required_inputs_num,
        size_t indices_idx,
        size_t per_sample_weights_idx,
        size_t default_index_idx,
        const std::set<Precision>& supported_precisions = {});

    StatusCode execute(
        std::vector<Blob::Ptr>& inputs,
        std::vector<Blob::Ptr>& outputs,
        ResponseDesc *resp) noexcept override;

protected:
    virtual void init_from_inputs(std::vector<Blob::Ptr>& inputs) = 0;
    virtual void get_indices(
        size_t emb_index,
        const size_t*& indices,
        size_t& size,
        size_t& weights_idx,
        bool& with_weights) = 0;

    template<typename T>
    void process_data(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs) noexcept;

    std::set<Precision> _supported_precisions;

    const size_t INDICES_IDX;
    const size_t PER_SAMPLE_WEIGHTS_IDX;
    const size_t DEFAULT_INDEX_IDX;

    bool _with_weights = false;
    size_t _embDepth = 0;
    std::string _l_name;

    using INT32 = PrecisionTrait<Precision::I32>::value_type;
    using INT64 = PrecisionTrait<Precision::I64>::value_type;
    using UINT64 = PrecisionTrait<Precision::U64>::value_type;

    static const std::set<size_t> _supported_indexes_type_size;
};

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
