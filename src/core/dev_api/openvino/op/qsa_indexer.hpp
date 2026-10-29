// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/op/op.hpp"

namespace ov::op::internal {

/// \brief Stateful QSA index preparation and block selection operation.
///
/// Inputs are index_query, raw_index_keys, rope_cos, rope_sin, key_length, q_norm_weight,
/// k_norm_weight, block_index_keys_cache, past_incomplete_raw_k, and past_incomplete_rope.
/// Outputs are selected block indices/count and the updated three state tensors.
class OPENVINO_API QSAIndexer : public ov::op::Op {
public:
    OPENVINO_OP("QSAIndexer");

    struct Config {
        int64_t key_block_size = 4;
        int64_t token_budget = 0;
        int64_t num_selection_heads = 1;
        float q_epsilon = 1e-6f;
        float k_epsilon = 1e-6f;
        bool is_causal = true;
    };

    /// Input layouts: index_query [batch, heads, sequence, head_size], raw_index_keys,
    /// block_index_keys_cache and past_incomplete_raw_k [batch, sequence, head_size],
    /// rope_cos / rope_sin [batch, sequence, rotary_dim], past_incomplete_rope [batch, 2 * rotary_dim].
    static constexpr int64_t query_rank = 4;
    static constexpr int64_t keys_rank = 3;
    static constexpr int64_t rope_rank = 3;
    static constexpr int64_t incomplete_rope_rank = 2;

    QSAIndexer() = default;
    QSAIndexer(const ov::OutputVector& args, const Config& config);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const Config& get_config() const {
        return m_config;
    }

private:
    Config m_config{};
};

}  // namespace ov::op::internal
