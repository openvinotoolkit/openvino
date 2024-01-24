// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/op/rope.hpp"

namespace cldnn {
using RoPE = ov::intel_gpu::op::RoPE;

/// @brief Rotary Position Embedding primitive
struct rope : public primitive_base<rope> {
    CLDNN_DECLARE_PRIMITIVE(rope);

    rope() : primitive_base("", {}) {}

    /// @brief Constructs rope primitive
    /// @param id This primitive id
    /// @param inputs Inputs primitive id
    /// @param config
    rope(const primitive_id& id,
         const std::vector<input_info>& inputs,
         const RoPE::Config& config,
         const padding& output_padding = padding())
        : primitive_base(id, inputs, {output_padding}),
          config(config) {}

    /// @brief
    RoPE::Config config;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, config.gather_position_arg_id);
        seed = hash_combine(seed, config.head_cnt);
        seed = hash_combine(seed, config.head_size);
        seed = hash_combine(seed, config.input_trans0213);
        seed = hash_combine(seed, config.is_chatglm);
        seed = hash_combine(seed, config.is_interleaved);
        seed = hash_combine(seed, config.is_qwen);
        seed = hash_combine(seed, config.rotary_ndims);
        seed = hash_combine(seed, config.slice_start);
        seed = hash_combine(seed, config.slice_stop);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const rope>(rhs);

        return config.gather_position_arg_id == rhs_casted.config.gather_position_arg_id; //TODO
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<rope>::save(ob);
        ob << config.gather_position_arg_id; //TODO
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<rope>::load(ib);
        ib >> config.gather_position_arg_id; //TODO
    }
};
}  // namespace cldnn
