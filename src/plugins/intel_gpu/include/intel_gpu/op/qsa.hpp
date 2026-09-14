// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/qsa.hpp"

namespace ov::intel_gpu::op {

/// @brief GPU-internal Qwen Sparse Attention (QSA) operation.
///
/// QSA fuses the indexer (block scoring + top-k selection) with the sparse main attention
/// that consumes the selected KV blocks. The op owns four CM kernel families:
///   Q0 qsa_kv_cache_update  - writes the externally projected K_cur/V_cur into the paged f16 cache
///   Q1 qsa_prepare          - indexer Q/K projection, Q RMSNorm + partial RoPE, raw-K append,
///                             completed-block FP32 mean pooling + block-key RMSNorm + block-start RoPE
///   Q2 qsa_score_topk       - block scoring (per-head ReLU before head reduction), top-k
///   Q3 qsa_sparse_attention - selected-block attention with online softmax and output gate
///
/// The op does NOT perform the main Q/K/V projection nor the output projection (`o_proj`);
/// those stay as regular graph operations. Its `mixed` input is the output of
/// GatedResidual READ/MIX, not the raw widened residual.
///
/// This class reuses ov::op::internal::QSA's Config / InputIdx / shape-inference / attribute
/// visiting logic wholesale (identical semantics); the only GPU-specific bit is the
/// "gpu_opset" RTTI tag consumed by ProgramBuilder's REGISTER_FACTORY dispatch and by
/// cldnn::qsa's primitive lowering (see intel_gpu/src/plugin/ops/qsa.cpp,
/// intel_gpu/src/graph/qsa.cpp).
class QSA : public ov::op::internal::QSA {
public:
    OPENVINO_OP("QSA", "gpu_opset", ov::op::internal::QSA);

    using Base = ov::op::internal::QSA;
    using Config = Base::Config;
    using InputIdx = Base::InputIdx;
    using Base::validate_config;

    QSA() = default;

    QSA(const ov::OutputVector& args, const Config& config, const ov::element::Type output_type = ov::element::dynamic)
        : Base(args, config, output_type) {}

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace ov::intel_gpu::op
