// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/sparse_pa.hpp"

namespace ov::intel_gpu::op {

/// @brief GPU-internal paged-cache sparse-attention consumer (Q0 cache write + Q3
/// selected-block attention). Reuses ov::op::internal::SparsePA's Config/InputIdx/
/// shape-inference/attribute-visiting logic wholesale.
class SparsePA : public ov::op::internal::SparsePA {
public:
    OPENVINO_OP("SparsePA", "gpu_opset", ov::op::internal::SparsePA);

    using Base = ov::op::internal::SparsePA;
    using Config = Base::Config;
    using InputIdx = Base::InputIdx;
    using Base::validate_config;

    SparsePA() = default;

    SparsePA(const ov::OutputVector& args, const Config& config, const ov::element::Type output_type = ov::element::dynamic)
        : Base(args, config, output_type) {}

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace ov::intel_gpu::op
