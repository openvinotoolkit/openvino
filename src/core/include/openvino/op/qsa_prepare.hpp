// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::v17 {
/// \brief Builds one index key per complete key block for sparse attention selection.
///
/// A block key is the mean of the raw index keys of the `key_block_size` consecutive key tokens of
/// the block, RMS-normalized with `k_norm_weight` and rotated with the RoPE of the first token of
/// the block. Only the blocks completed by the new tokens are computed: the keys of the earlier
/// blocks are passed in through `past_block_index_keys`, and the raw keys and the RoPE row of the
/// tokens of the still incomplete block through `past_incomplete_raw_keys` / `past_incomplete_rope`,
/// which the operation returns updated for the next call. The result feeds SparseAttentionIndexer.
///
/// Inputs are positional and cannot be skipped; a feature is disabled by connecting an empty constant.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API QSAPrepare : public Op {
public:
    OPENVINO_OP("QSAPrepare", "opset17", Op);

    QSAPrepare() = default;

    /// \brief Constructs a QSAPrepare operation from an ordered list of inputs.
    ///
    /// \param inputs raw_index_keys, past_block_index_keys, key_length, rope_cos, rope_sin,
    ///               k_norm_weight, past_incomplete_raw_keys and past_incomplete_rope
    /// \param key_block_size Number of consecutive key tokens forming one key block
    /// \param epsilon Value added to the mean square in the RMS normalization
    QSAPrepare(const OutputVector& inputs, int64_t key_block_size, float epsilon = 1e-6f);

    /// \brief Constructs a QSAPrepare operation.
    ///
    /// \param raw_index_keys Raw index key of every new token, shape [N, L, Di]
    /// \param past_block_index_keys Block index keys computed by the previous calls, shape [N, Bp, Di]
    /// \param key_length Number of valid key tokens of every sequence, shape [N] or [1]
    /// \param rope_cos RoPE cos of every new token, shape [N, L, Dr], empty constant to disable the rotation
    /// \param rope_sin RoPE sin of every new token, shape [N, L, Dr], empty constant to disable the rotation
    /// \param k_norm_weight Scale of the block key normalization, shape [Di], empty constant to disable it
    /// \param past_incomplete_raw_keys Raw keys of the past tokens of the incomplete block,
    ///                                 shape [N, key_block_size - 1, Di]
    /// \param past_incomplete_rope RoPE cos and sin of the first token of the incomplete block,
    ///                             shape [N, 2 * Dr], empty constant if and only if rope_cos and rope_sin are
    QSAPrepare(const Output<Node>& raw_index_keys,
               const Output<Node>& past_block_index_keys,
               const Output<Node>& key_length,
               const Output<Node>& rope_cos,
               const Output<Node>& rope_sin,
               const Output<Node>& k_norm_weight,
               const Output<Node>& past_incomplete_raw_keys,
               const Output<Node>& past_incomplete_rope,
               int64_t key_block_size,
               float epsilon = 1e-6f);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    int64_t get_key_block_size() const {
        return m_key_block_size;
    }
    void set_key_block_size(int64_t key_block_size) {
        m_key_block_size = key_block_size;
    }

    float get_epsilon() const {
        return m_epsilon;
    }
    void set_epsilon(float epsilon) {
        m_epsilon = epsilon;
    }

private:
    int64_t m_key_block_size = 0;
    float m_epsilon = 1e-6f;
};
}  // namespace ov::op::v17
