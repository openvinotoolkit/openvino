// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base.hpp"
#include "jit_generator.hpp"

#include <memory>
#include <set>
#include <vector>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

struct jit_emb_bag_config_params {
    size_t emb_dim;
    bool with_weights; // Initialized on creation dependent on weights input existence.
};

struct jit_emb_bag_call_args {
    const void* src;
    void* dst;
    const void* weights;
    size_t with_weights; // Need to detect empty bag. They are not multiplied on weights.
};

struct jit_uni_embedding_bag_sum_kernel {
    void (*_ker)(const jit_emb_bag_call_args *);

    void operator()(const jit_emb_bag_call_args *args) {
        assert(_ker);
        _ker(args);
    }

    explicit jit_uni_embedding_bag_sum_kernel(jit_emb_bag_config_params jcp) : _ker(nullptr), _jcp(jcp) {}
    virtual ~jit_uni_embedding_bag_sum_kernel() {}

protected:
    jit_emb_bag_config_params _jcp;

    Xbyak::Reg64 reg_src = Xbyak::util::r8;
    Xbyak::Reg64 reg_dst = Xbyak::util::r9;
    Xbyak::Reg64 reg_weights = Xbyak::util::r10;
    Xbyak::Reg64 reg_params = mkldnn::impl::cpu::abi_param1;
};

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

    size_t _default_index = 0lu;
    bool _with_weights = false;
    std::vector<size_t> _multipliers;
    std::string _l_name;

    std::shared_ptr<jit_uni_embedding_bag_sum_kernel> emb_bag_kernel;

    using INT32 = PrecisionTrait<Precision::I32>::value_type;
    using INT64 = PrecisionTrait<Precision::I64>::value_type;
    using UINT64 = PrecisionTrait<Precision::U64>::value_type;

    static const std::set<size_t> _supported_indexes_type_size;
};

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
