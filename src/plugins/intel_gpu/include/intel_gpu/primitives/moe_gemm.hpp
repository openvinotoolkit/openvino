// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/op/moe_compressed.hpp"

namespace cldnn {

/// @brief    gemm for moe_pattern which selectively executes experts.
/// @details
/// This primitive implements the GEMM operation for the Mixture of Experts (MoE) pattern,
/// allowing for efficient execution of a subset of experts based on the input data.
/// @param input         Input data tensor.
/// @param weight        Weights tensor containing expert weights.
/// @param experts_ids   Tensor containing the IDs of the experts that are actually used at each time.
/// @param inputs_offset_per_expert   Tensor containing the offsets information of inputs per expert.
/// @param input_tokens_lens   Tensor containing the lengths of input tokens used by each expert.
/// @param num_experts_per_token  Number of experts per token selected by router.
struct moe_gemm : public primitive_base<moe_gemm> {
    CLDNN_DECLARE_PRIMITIVE(moe_gemm)

    enum MoEGemmInputIdx {
        // required
        INPUT = 0,
        WEIGHT = 1,
        EXPERTS_IDS = 2,
        INPUT_OFFSET_PER_EXPERT = 3,
        INPUT_TOKENS_LENS = 4,
        // optional
        BIAS = 5,
        WEIGHT_SCALE = 6,
        WEIGHT_ZP = 7
    };

    moe_gemm() : primitive_base("", {}) {}

    /// @brief Constructs moe_gemm primitive.
    ///
    moe_gemm(const primitive_id& id,
             const std::vector<input_info>& inputs,
             const ov::intel_gpu::op::MOECompressed::Config& moe_config)
          : primitive_base(id, inputs),
            num_experts_per_token(static_cast<int32_t>(moe_config.top_k)),
            has_batch_dim(moe_config.has_batch_dim) {}

    bool has_bias = false;
    int32_t num_experts_per_token = 0;
    bool has_batch_dim = true;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, has_bias);
        seed = hash_combine(seed, num_experts_per_token);
        seed = hash_combine(seed, has_batch_dim);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        auto rhs_casted = downcast<const moe_gemm>(rhs);
        return has_bias == rhs_casted.has_bias &&
               num_experts_per_token == rhs_casted.num_experts_per_token &&
               has_batch_dim == rhs_casted.has_batch_dim;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<moe_gemm>::save(ob);
        ob << has_bias;
        ob << num_experts_per_token;
        ob << has_batch_dim;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<moe_gemm>::load(ib);
        ib >> has_bias;
        ib >> num_experts_per_token;
        ib >> has_batch_dim;
    }
};
}
