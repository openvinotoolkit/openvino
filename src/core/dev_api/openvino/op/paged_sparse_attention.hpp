// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>

#include "openvino/op/op.hpp"

namespace ov::op::internal {

/// \brief Causal sparse attention over a paged key/value cache.
///
/// The positional inputs are query, key, value, key_cache, value_cache, past_lens,
/// subsequence_begins, block_indices, block_indices_begins, selected_block_indices,
/// selected_block_count, scale, and sinks. The key/value cache inputs are updated in place.
class OPENVINO_API PagedSparseAttention : public Op {
public:
    OPENVINO_OP("PagedSparseAttention");

    PagedSparseAttention() = default;

    /// \brief Constructs a PagedSparseAttention operation.
    /// \param inputs The 13 positional inputs defined by the operation specification.
    /// \param key_block_size Number of consecutive key tokens in one selected block.
    PagedSparseAttention(const OutputVector& inputs, int64_t key_block_size);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    int64_t get_key_block_size() const {
        return m_key_block_size;
    }

    void set_key_block_size(int64_t key_block_size) {
        m_key_block_size = key_block_size;
    }

private:
    int64_t m_key_block_size = 1;
};

}  // namespace ov::op::internal
