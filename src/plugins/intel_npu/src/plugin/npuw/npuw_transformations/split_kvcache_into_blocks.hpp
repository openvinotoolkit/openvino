// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace npuw {
namespace pass {

/**
 * @brief Transformation pass to split KV cache access into block-based pattern
 *
 * This pass identifies KV cache parameters in the model and transforms their access
 * pattern from continuous memory (batch, num_heads, seq_len, head_dim) to block-based
 * layout (num_blocks, batch, num_heads, block_size, head_dim).
 *
 * Transformation pattern:
 * Before:
 *   Parameter(past_key)     [1, 32, 2048, 128]  \
 *   Parameter(present_key)  [1, 32, 1, 128]      |-> Concat(axis=2) -> SDPA
 *
 *   Parameter(past_value)   [1, 32, 128, 2048]  \
 *   Parameter(present_value)[1, 32, 128, 1]      |-> Concat(axis=3) -> SDPA  (if v_transposed=true)
 *
 * After (block_size=1024, seq_len=2048):
 *   Parameter(past_key_block_0)   [1, 32, 1024, 128]  \
 *   Parameter(past_key_block_1)   [1, 32, 1024, 128]   |
 *   Parameter(present_key)        [1, 32, 1, 128]      |-> Concat(axis=2) -> SDPA
 *
 *   Parameter(past_value_block_0) [1, 32, 128, 1024]  \
 *   Parameter(past_value_block_1) [1, 32, 128, 1024]   |
 *   Parameter(present_value)      [1, 32, 128, 1]      |-> Concat(axis=3) -> SDPA  (if v_transposed=true)
 *
 * Note: Number of blocks is auto-calculated from original seq_len and block_size.
 *       If seq_len % block_size != 0, a tail block with remaining tokens is created.
 *
 * Benefits:
 * - Reduces memory allocation (only allocate needed blocks)
 * - Enables memory sharing across sequences
 * - Improves memory locality for attention operations
 *
 * Note: This transformation is applied to both prefill and generate (decode) models
 * where KV cache parameters are identified by naming convention
 * (e.g., "past_key", "past_value", "past_key_values.X.key", "past_key_values.X.value").
 */
class SplitKVCacheIntoBlocks : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ov::npuw::pass::SplitKVCacheIntoBlocks", "0");

    /**
     * @brief Construct transformation with block configuration
     *
     * @param block_size Number of tokens per block (default: 1024 for efficiency)
     * @param v_transposed Whether V tensor is transposed (true: [B,H,D,S], false: [B,H,S,D])
     *
     * The number of blocks is automatically calculated from the original past_key shape.
     * If the sequence length is not evenly divisible by block_size, a tail block is created.
     */
    explicit SplitKVCacheIntoBlocks(uint32_t block_size = 1024, bool v_transposed = true);

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    uint32_t m_block_size;
    bool m_v_transposed;
};

}  // namespace pass
}  // namespace npuw
}  // namespace ov
