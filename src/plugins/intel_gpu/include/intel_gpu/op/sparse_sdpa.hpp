// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/sparse_sdpa.hpp"

namespace ov::intel_gpu::op {

/// @brief GPU-internal non-paged (dense KVCache-managed) sparse-attention consumer (Q3
/// only, read-only on main K/V). Reuses ov::op::internal::SparseSDPA's Config/InputIdx/
/// shape-inference/attribute-visiting logic wholesale.
class SparseSDPA : public ov::op::internal::SparseSDPA {
public:
    OPENVINO_OP("SparseSDPA", "gpu_opset", ov::op::internal::SparseSDPA);

    using Base = ov::op::internal::SparseSDPA;
    using Config = Base::Config;
    using InputIdx = Base::InputIdx;
    using Base::validate_config;

    SparseSDPA() = default;

    SparseSDPA(const ov::OutputVector& args, const Config& config, const ov::element::Type output_type = ov::element::dynamic)
        : Base(args, config, output_type) {}

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace ov::intel_gpu::op
