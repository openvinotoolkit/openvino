// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/qsa_sdpa.hpp"

namespace ov::intel_gpu::op {

/// @brief GPU-internal SDPA-style (dense, KVCache-managed) Qwen Sparse Attention operation.
///
/// See ov::op::internal::QSASDPA for the full description; this class reuses its Config /
/// InputIdx / shape-inference / attribute-visiting logic wholesale (identical semantics),
/// the only GPU-specific bit is the "gpu_opset" RTTI tag consumed by ProgramBuilder's
/// REGISTER_FACTORY dispatch and by cldnn::qsa's primitive lowering (see
/// intel_gpu/src/plugin/ops/qsa_sdpa.cpp, intel_gpu/src/graph/qsa.cpp).
class QSASDPA : public ov::op::internal::QSASDPA {
public:
    OPENVINO_OP("QSASDPA", "gpu_opset", ov::op::internal::QSASDPA);

    using Base = ov::op::internal::QSASDPA;
    using Config = Base::Config;
    using InputIdx = Base::InputIdx;
    using Base::validate_config;

    QSASDPA() = default;

    QSASDPA(const ov::OutputVector& args, const Config& config, const ov::element::Type output_type = ov::element::dynamic)
        : Base(args, config, output_type) {}

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace ov::intel_gpu::op
