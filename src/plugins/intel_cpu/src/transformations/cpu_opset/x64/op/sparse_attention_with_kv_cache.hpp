// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/op.hpp"

namespace ov::intel_cpu {

/// \brief ov::op::v17::SparseAttention fused with the KV cache update (Gather(beam) + Concat + Assign).
///
/// Counterpart of ScaledDotProductAttentionWithKVCache (fuse_concat == true) for the sparse attention op.
/// The key/value cache lives in the plugin's KV cache variable states instead of in a Concat output, so the
/// cache is appended in place and never copied as a whole on every step.
///
/// Inputs (the last three are the stateful part, always in this order):
///   0 query            [N, H, L, E]
///   1 cur_key          [N, Hkv, L, E]    current step key,   the cache is cat(past_key, cur_key) along axis 2
///   2 cur_value        [N, Hkv, L, Ev]   current step value, the cache is cat(past_value, cur_value) along axis 2
///   3 block_indices    [N, Hs, L, KB]
///   4 block_count      [N, Hs, L]
///   5 key_length       [N] or [1], optional (the 8-input form has no key_length)
///   n-3 beam_idx       [N]
///   n-2 past_key       [N, Hkv, L0, E]   output of the ReadValue of the key state
///   n-1 past_value     [N, Hkv, L0, Ev]  output of the ReadValue of the value state
///
/// Outputs: 0 attention [N, H, L, Ev]; 1 present_key [N, Hkv, L0 + L, E] and 2 present_value
/// [N, Hkv, L0 + L, Ev], to be consumed by the Assign nodes of the key/value states.
///
/// \ingroup ov_ops_cpp_api
class SparseAttentionWithKVCache : public ov::op::Op {
public:
    OPENVINO_OP("SparseAttentionWithKVCache", "cpu_plugin_opset");

    /// @brief Fixed positions of the leading inputs; the state inputs are addressed from the end.
    struct InputIdx {
        static constexpr size_t QUERY = 0;
        static constexpr size_t CUR_KEY = 1;
        static constexpr size_t CUR_VALUE = 2;
        static constexpr size_t BLOCK_INDICES = 3;
        static constexpr size_t BLOCK_COUNT = 4;
        static constexpr size_t KEY_LENGTH = 5;  ///< optional
        static constexpr size_t COUNT_MIN = 5 + 3;
        static constexpr size_t COUNT_MAX = 6 + 3;
    };

    struct Config {
        int64_t key_block_size = 1;
        int64_t query_block_size = 1;
        bool causal = true;
    };

    SparseAttentionWithKVCache() = default;
    SparseAttentionWithKVCache(const OutputVector& args, const Config& cfg);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    const Config& get_config() const {
        return m_config;
    }

    [[nodiscard]] bool has_key_length() const {
        return get_input_size() == InputIdx::COUNT_MAX;
    }

private:
    Config m_config;
};

}  // namespace ov::intel_cpu
