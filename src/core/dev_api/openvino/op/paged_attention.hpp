// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {

/// \brief PagedAttentionExtension operation implements paged attention for memory-efficient sequence processing.
///
/// \ingroup ov_ops_cpp_api
///
/// This operation computes attention using a paged memory model, allowing efficient handling of long sequences.

class OPENVINO_API PagedAttentionExtension : public ov::op::Op {
public:
    OPENVINO_OP("PagedAttentionExtension");

    PagedAttentionExtension() = default;

    /// \brief Constructs a PagedAttentionExtension operation.
    ///
    /// \param args Input arguments vector containing:
    //  0  query
    //  1  key
    //  2  value
    //  3  key_cache (initial KV-cache content)
    //  4  value_cache (initial KV-cache content)
    //  5  past_lens
    //  6  subsequence_begins
    //  7  block_indices (initial block table)
    //  8  block_indices_begins (initial block table offsets)
    //  9  scale
    // 10  sliding_window
    // 11  alibi_slopes
    // 12  max_context_len
    // 13  score_aggregation_window
    // 14  rotated_block_indices
    // 15  rotation_deltas
    // 16  rotation_trig_lut
    // 17  xattention_threshold
    // 18  xattention_block_size
    // 19  xattention_stride
    // 20  sinks
    // 21  adaptive_rkv_start_size
    // 22  adaptive_rkv_evictable_sizes
    // 23  adaptive_rkv_diversity_block_set_indices
    // 24  adaptive_rkv_diversity_block_set_indices_begins
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
