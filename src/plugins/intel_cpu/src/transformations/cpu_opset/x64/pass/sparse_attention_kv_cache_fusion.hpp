// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_set>

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu {

/// \brief Fuses the KV cache update feeding ov::op::v17::SparseAttention into one SparseAttentionWithKVCache op:
///
///   ReadValue -> [Convert] -> Gather(beam_idx) -> Concat(cur) -> SparseAttention (key, value inputs)
///                                                      \-> [Convert] -> Assign
///
/// This is the sparse counterpart of StatefulSDPAFusion. The fused op keeps the cache in the plugin's KV cache
/// states (see MatchSparseSdpaKvCache), so the history is not copied by the Concat on every step. Only the
/// canonical layout is handled: key/value are [N, Hkv, S, E] and are concatenated along axis 2.
class SparseAttentionKVCacheFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SparseAttentionKVCacheFusion");
    SparseAttentionKVCacheFusion();

private:
    // Variable ids whose SparseAttention has already been fused; later matches on the same Variable are
    // reader-only consumers and must be skipped.
    std::unordered_set<std::string> m_processed_k_variable_ids;
    std::unordered_set<std::string> m_processed_v_variable_ids;
};

}  // namespace ov::intel_cpu
