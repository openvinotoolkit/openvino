// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {

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
    ///  0  query                                         [B_token, H * S]
    ///  1  key                                           [B_token, Hk * S]
    ///  2  value                                         [B_token, Hk * S]
    ///  3  key_cache                                     [num_blocks, Hk, block_size, S]
    ///  4  value_cache                                   [num_blocks, Hk, block_size, S]
    ///  5  past_lens                                     [B_seq], i32
    ///  6  subsequence_begins                            [B_seq + 1], i32
    ///  7  block_indices                                 [total_blocks], i32
    ///  8  block_indices_begins                          [B_seq + 1], i32
    ///  9  scale                                         [] scalar
    /// 10  sliding_window                                [] scalar, i32
    /// 11  alibi_slopes                                  [H] or empty
    /// 12  max_context_len                               [] scalar, i32
    /// 13  score_aggregation_window                      [] scalar or [B_seq], i32
    /// 14  rotated_block_indices                         [num_rotated_blocks] or empty, i32
    /// 15  rotation_deltas                               [num_rotated_blocks] or [num_rotated_blocks, block_size], i32
    /// 16  rotation_trig_lut                             [max_context_len * S] or [max_context_len, S]
    /// 17  xattention_threshold                          [B_seq, H] or empty
    /// 18  xattention_block_size                         [] scalar, i32
    /// 19  xattention_stride                             [] scalar, i32
    /// 20  sinks                                         [1, H, 1, 1] or empty
    /// 21  adaptive_rkv_start_size                       [] scalar, i32
    /// 22  adaptive_rkv_evictable_sizes                  [B_seq], i32
    /// 23  adaptive_rkv_diversity_block_set_indices      [num_adaptive_rkv_blocks], i32
    /// 24  adaptive_rkv_diversity_block_set_indices_begins  [B_seq + 1], i32
    explicit PagedAttentionExtension(const ov::OutputVector& args);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const ov::element::Type get_out_type(int index) const;
    void set_out_type(int index, const ov::element::Type& output_type);

    using PagedCacheManagerHandle = std::shared_ptr<void>;  // Void handle to avoid inconsistent linkage
    PagedCacheManagerHandle get_cache_manager() const;
    void set_cache_manager(PagedCacheManagerHandle cache_manager);

protected:
    PagedCacheManagerHandle m_cache_manager = nullptr;
    std::vector<ov::element::Type> m_output_type = {ov::element::dynamic, ov::element::dynamic, ov::element::dynamic};
};

// Exported function for transformations to construct the manager; avoids C4273 in some build configs
OPENVINO_API PagedAttentionExtension::PagedCacheManagerHandle make_paged_cache_handle(ov::element::Type et);

}  // namespace op
}  // namespace ov
