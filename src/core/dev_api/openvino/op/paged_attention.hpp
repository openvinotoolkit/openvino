// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {

// This is an experimental operation that is implemented in the plugins.
// Do not use in user applications, backward compatibility is not guaranteed in future releases.

/// \brief PagedAttentionExtension operation implements paged attention for memory-efficient sequence processing
///
/// \ingroup ov_ops_cpp_api
///
/// This operation computes attention using a paged memory model, allowing efficient handling of long sequences
class OPENVINO_API PagedAttentionExtension : public ov::op::Op {
public:
    OPENVINO_OP("PagedAttentionExtension");

    PagedAttentionExtension() = default;

    /// \brief Constructs a PagedAttentionExtension operation
    ///
    /// \param args Input arguments vector containing:
    ///   (B_token = total tokens in the call, B_seq = number of sequences,
    ///    H = query heads, Hk = key/value heads, S = head size)
    ///
    ///  0  query                                            [B_token, H * S]          required
    ///  1  key                                              [B_token, Hk * S]         required
    ///  2  value                                            [B_token, Hk * S]         required
    ///  3  key_cache                                        [num_blocks, Hk, Bs, S]   required
    ///  4  value_cache                                      [num_blocks, Hk, Bs, S]   required
    ///  5  past_lens                                        [B_seq], i32              required
    ///  6  subsequence_begins                               [B_seq + 1], i32          required
    ///  7  block_indices                                    [total_blocks], i32        required
    ///  8  block_indices_begins                             [B_seq + 1], i32          required
    ///  9  scale                                            [] scalar                 required
    /// 10  sliding_window                                   [] scalar, i32            required (0 = unlimited)
    /// 11  alibi_slopes                                     [H] or empty              required (empty = disabled)
    /// 12  max_context_len                                  [] scalar, i32            required (0 = unlimited)
    /// 13  score_aggregation_window                         [] scalar or [B_seq], i32 required (0 = disabled)
    /// 14  rotated_block_indices                            [Nrot] or empty, i32      required (empty = disabled)
    /// 15  rotation_deltas                                  [Nrot] or [Nrot, Bs], i32 required (empty = disabled)
    /// 16  rotation_trig_lut                                [C, S] or [C*S]           required (empty = disabled)
    /// 17  xattention_threshold                             [] or [B_seq]             required (empty = disabled)
    /// 18  xattention_block_size                            [] scalar, i32            required (0 = disabled)
    /// 19  xattention_stride                                [] scalar, i32            required
    /// 20  sinks                                            [1, H, 1, 1] or empty     required (empty = disabled)
    /// 21  adaptive_rkv_start_size                          [] scalar, i32            required (0 = no protection zone)
    /// 22  adaptive_rkv_evictable_sizes                     [B_seq], i32              optional
    /// 23  adaptive_rkv_diversity_block_set_indices         [num_adaptive_rkv_blocks] optional
    /// 24  adaptive_rkv_diversity_block_set_indices_begins  [B_seq + 1], i32          optional
    explicit PagedAttentionExtension(const ov::OutputVector& args);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const ov::element::Type get_out_type(int index) const;
    void set_out_type(int index, const ov::element::Type& output_type);

protected:
    std::vector<ov::element::Type> m_output_type = {ov::element::dynamic, ov::element::dynamic, ov::element::dynamic};
};

}  // namespace op
}  // namespace ov
