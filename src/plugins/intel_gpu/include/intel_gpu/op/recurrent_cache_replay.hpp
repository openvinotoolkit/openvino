// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include "openvino/op/op.hpp"

namespace ov::intel_gpu::op {
// GPU-internal lowering of the llama.cpp OpenVINO backend's recurrent replay graphs.
// The backend emits a standard-op computation and Assign sinks with variable IDs
// `__ggml_replay_cache_<Parameter friendly name>`. Each named bank is a direct,
// caller-owned writable FP32 USM input. RecurrentReplayFusion checks the full pattern,
// exclusive bank readers and a live compute result before removing the sinks.
// Both ops return the computation and update the bank inputs in place, avoiding
// special output-alias handling in the GPU runtime. If the backend's emitted graph
// changes, update the matcher and its reference in recurrent_replay_graph.hpp together.

// Returns the convolution and updates the caller-owned USM history input in place.
class RecurrentConvCacheReplayUpdate : public ov::op::Op {
  public:
    OPENVINO_OP("RecurrentConvCacheReplayUpdate", "gpu_opset");
    RecurrentConvCacheReplayUpdate() = default;
    RecurrentConvCacheReplayUpdate(const OutputVector& args, int64_t history);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor&) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector&) const override;
    int64_t get_history() const { return m_history; }

  private:
    int64_t m_history = 0;
};

// State and the three pending banks are updated in place. With keep=true the
// committed state stops after replay, while current-token attention uses the
// subsequent working state. Current prepared K/V/gate/beta prefixes are saved
// in all cases; untouched destination tails and beta padding are preserved.
class RecurrentGatedDeltaNetReplayUpdate : public ov::op::Op {
  public:
    OPENVINO_OP("RecurrentGatedDeltaNetReplayUpdate", "gpu_opset");
    RecurrentGatedDeltaNetReplayUpdate() = default;
    RecurrentGatedDeltaNetReplayUpdate(const OutputVector& args, bool fuse_qk_l2norm = false,
                                       float q_l2_norm_eps = 1e-6f, float k_l2_norm_eps = 1e-6f,
                                       std::string head_mapping = "tiled");
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor&) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector&) const override;
    bool get_fuse_qk_l2norm() const { return m_fuse_qk_l2norm; }
    float get_q_l2_norm_eps() const { return m_q_l2_norm_eps; }
    float get_k_l2_norm_eps() const { return m_k_l2_norm_eps; }
    const std::string& get_head_mapping() const { return m_head_mapping; }

  private:
    bool m_fuse_qk_l2norm = false;
    float m_q_l2_norm_eps = 1e-6f;
    float m_k_l2_norm_eps = 1e-6f;
    std::string m_head_mapping = "tiled";
};
} // namespace ov::intel_gpu::op
