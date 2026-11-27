// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/op/op.hpp"
#include "openvino/op/qsa_indexer.hpp"

namespace ov::intel_gpu::op {

/// \brief ov::op::internal::QSAIndexer whose block index key cache is a stateful KV cache updated in place.
///
/// Inputs and attributes are those of QSAIndexer, except that BLOCK_INDEX_KEYS_CACHE is the [B, blocks, D_idx] output of
/// the KVCache that holds the cache variable (one index key head, so no head axis), with at least
/// floor(max(key_length) / key_block_size) blocks. The operation writes the blocks the call completes into that buffer
/// instead of returning an updated copy, so output BLOCK_INDEX_KEYS_CACHE is an empty placeholder that only keeps the
/// QSAIndexer output order. The other outputs are those of QSAIndexer.
class QSAIndexerInPlace : public ov::op::Op {
public:
    OPENVINO_OP("QSAIndexerInPlace", "gpu_opset");

    using InputIdx = ov::op::internal::QSAIndexer::InputIdx;
    using OutputIdx = ov::op::internal::QSAIndexer::OutputIdx;
    using Config = ov::op::internal::QSAIndexer::Config;

    QSAIndexerInPlace() = default;
    QSAIndexerInPlace(const ov::OutputVector& args, const Config& config);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const Config& get_config() const {
        return m_config;
    }

private:
    Config m_config{};
};

}  // namespace ov::intel_gpu::op
