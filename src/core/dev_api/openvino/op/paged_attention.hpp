// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/paged_cache_manager.hpp"
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
    ///             - query
    ///             - key
    ///             - value
    ///             - key_cache
    ///             - value_cache
    ///             - past_lens
    ///             - subsequence_begins
    ///             - block_indices
    ///             - block_indices_begins
    ///             - (optional) scale
    ///             - (optional) sliding_window
    ///             - (optional) alibi_slopes
    ///             - max_context_len
    ///             - (optional) rotated_block_indices
    ///             - (optional) rotation_deltas
    ///             - (optional) rotation_trig_lut
    PagedAttentionExtension(const ov::OutputVector& args);

    /// \brief Constructs a PagedAttentionExtension operation. (13 parameter constructor)
    ///
    /// \param query                Query tensor.
    /// \param key                  Key tensor.
    /// \param value                Value tensor.
    /// \param key_cache            Cached key tensor.
    /// \param value_cache          Cached value tensor.
    /// \param past_lens            Lengths of past sequences.
    /// \param subsequence_begins   Subsequence start indices.
    /// \param block_indices        Indices of memory blocks.
    /// \param block_indices_begins Start indices for block indexing.
    /// \param scale                (Optional) Scaling factor for attention scores.
    /// \param sliding_window       (Optional) Sliding window size for local attention.
    /// \param alibi_slopes         (Optional) ALiBi slopes for biasing attention.
    /// \param max_context_len      Maximum context length.
    PagedAttentionExtension(const Output<Node>& query,
                            const Output<Node>& key,
                            const Output<Node>& value,
                            const Output<Node>& key_cache,
                            const Output<Node>& value_cache,
                            const Output<Node>& past_lens,
                            const Output<Node>& subsequence_begins,
                            const Output<Node>& block_indices,
                            const Output<Node>& block_indices_begins,
                            const Output<Node>& scale,
                            const Output<Node>& sliding_window,
                            const Output<Node>& alibi_slopes,
                            const Output<Node>& max_context_len);

    /// \brief Constructs a PagedAttentionExtension operation with rotation support. (16 parameter constructor)
    ///
    /// \param query                Query tensor.
    /// \param key                  Key tensor.
    /// \param value                Value tensor.
    /// \param key_cache            Cached key tensor.
    /// \param value_cache          Cached value tensor.
    /// \param past_lens            Lengths of past sequences.
    /// \param subsequence_begins   Subsequence start indices.
    /// \param block_indices        Indices of memory blocks.
    /// \param block_indices_begins Start indices for block indexing.
    /// \param scale                (Optional) Scaling factor for attention scores.
    /// \param sliding_window       (Optional) Sliding window size for local attention.
    /// \param alibi_slopes         (Optional) ALiBi slopes for biasing attention.
    /// \param max_context_len      Maximum context length.
    /// \param rotated_block_indices (Optional) Rotated block indices.
    /// \param rotation_deltas       (Optional) Rotation deltas.
    /// \param rotation_trig_lut     (Optional) Rotation trig lookup table.
    PagedAttentionExtension(const Output<Node>& query,
                            const Output<Node>& key,
                            const Output<Node>& value,
                            const Output<Node>& key_cache,
                            const Output<Node>& value_cache,
                            const Output<Node>& past_lens,
                            const Output<Node>& subsequence_begins,
                            const Output<Node>& block_indices,
                            const Output<Node>& block_indices_begins,
                            const Output<Node>& scale,
                            const Output<Node>& sliding_window,
                            const Output<Node>& alibi_slopes,
                            const Output<Node>& max_context_len,
                            const Output<Node>& rotated_block_indices,
                            const Output<Node>& rotation_deltas,
                            const Output<Node>& rotation_trig_lut);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    /// \brief Gets the output element type at the specified index.
    const ov::element::Type get_out_type(int index) const;

    /// \brief Sets the output element type at the specified index.
    void set_out_type(int index, const ov::element::Type& output_type);

    const std::shared_ptr<ov::internal::PagedCacheManager> get_cache_manager() const;

    void set_cache_manager(const std::shared_ptr<ov::internal::PagedCacheManager> cache_manager);

protected:
    std::vector<ov::element::Type> m_output_type = {ov::element::dynamic, ov::element::dynamic};
    std::shared_ptr<ov::internal::PagedCacheManager> m_cache_manager = nullptr;
};

}  // namespace op
}  // namespace ov
