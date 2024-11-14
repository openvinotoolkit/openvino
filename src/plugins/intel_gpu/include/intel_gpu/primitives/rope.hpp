// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

namespace cldnn {
using RoPE = ov::op::internal::RoPE;

/// @brief Rotary Position Embedding primitive
struct rope : public primitive_base<rope> {
    CLDNN_DECLARE_PRIMITIVE(rope);

    rope() : primitive_base("", {}) {}

    /// @brief Constructs rope primitive
    /// @param id This primitive id
    /// @param inputs Inputs primitive ids
    /// @param config Specific RoPE config
    /// @param gather_rank Required for correct processing of gather input (if applicable)
    rope(const primitive_id& id,
         const std::vector<input_info>& inputs,
         const RoPE::Config& config,
         size_t gather_rank = 0)
        : primitive_base(id, inputs),
          config(config),
          gather_rank(gather_rank) {
            OPENVINO_ASSERT((!config.support_2d_rope
                || (config.support_2d_rope && config.is_chatglm)),
                "2D RoPE is currently only supported in Chatglm!");
        }

    RoPE::Config config;
    size_t gather_rank = 0;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, config.gather_position_arg_id);
        seed = hash_combine(seed, config.head_cnt);
        seed = hash_combine(seed, config.head_size);
        seed = hash_combine(seed, config.input_trans0213);
        seed = hash_combine(seed, config.is_chatglm);
        seed = hash_combine(seed, config.support_2d_rope);
        seed = hash_combine(seed, config.is_interleaved);
        seed = hash_combine(seed, config.is_qwen);
        seed = hash_combine(seed, config.rotary_ndims);
        seed = hash_combine(seed, config.slice_start);
        seed = hash_combine(seed, config.slice_stop);
        seed = hash_combine(seed, gather_rank);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const rope>(rhs);

        return config.gather_position_arg_id == rhs_casted.config.gather_position_arg_id &&
               config.head_cnt == rhs_casted.config.head_cnt &&
               config.head_size == rhs_casted.config.head_size &&
               config.input_trans0213 == rhs_casted.config.input_trans0213 &&
               config.is_chatglm == rhs_casted.config.is_chatglm &&
               config.support_2d_rope == rhs_casted.config.support_2d_rope &&
               config.is_interleaved == rhs_casted.config.is_interleaved &&
               config.is_qwen == rhs_casted.config.is_qwen &&
               config.rotary_ndims == rhs_casted.config.rotary_ndims &&
               config.slice_start == rhs_casted.config.slice_start &&
               config.slice_stop == rhs_casted.config.slice_stop &&
               gather_rank == rhs_casted.gather_rank;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<rope>::save(ob);
        ob << config.gather_position_arg_id;
        ob << config.head_cnt;
        ob << config.head_size;
        ob << config.input_trans0213;
        ob << config.is_chatglm;
        ob << config.support_2d_rope;
        ob << config.is_interleaved;
        ob << config.is_qwen;
        ob << config.rotary_ndims;
        ob << config.slice_start;
        ob << config.slice_stop;
        ob << gather_rank;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<rope>::load(ib);
        ib >> config.gather_position_arg_id;
        ib >> config.head_cnt;
        ib >> config.head_size;
        ib >> config.input_trans0213;
        ib >> config.is_chatglm;
        ib >> config.support_2d_rope;
        ib >> config.is_interleaved;
        ib >> config.is_qwen;
        ib >> config.rotary_ndims;
        ib >> config.slice_start;
        ib >> config.slice_stop;
        ib >> gather_rank;
    }
};
}  // namespace cldnn
