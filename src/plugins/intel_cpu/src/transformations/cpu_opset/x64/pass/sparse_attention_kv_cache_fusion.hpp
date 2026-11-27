// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_cpu {

/// \brief Fuses the KV cache update feeding ov::op::v17::SparseAttention into one SparseAttentionWithKVCache op:
///
///   ReadValue -> [Convert] -> Gather(beam_idx) -> Concat(cur) -> SparseAttention (key, value inputs)
///                                                      |-> [Convert] -> Assign
///                                                      \-> ShapeOf (optional)
///
/// This is the sparse counterpart of StatefulSDPAFusion. The fused op keeps the cache in the plugin's KV cache
/// states (see MatchSparseSdpaKvCache), so the history is not copied by the Concat on every step. Only the
/// canonical layout is handled: key/value are [N, Hkv, S, E] and are concatenated along axis 2 (or -2).
///
/// It is a ModelPass because the fusion replaces the outputs of the Concats for consumers outside of the matched
/// pattern (the Assign sinks of the states and the ShapeOf), and it remembers the fused Variables across the matches.
class SparseAttentionKVCacheFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("SparseAttentionKVCacheFusion");
    SparseAttentionKVCacheFusion() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::intel_cpu
