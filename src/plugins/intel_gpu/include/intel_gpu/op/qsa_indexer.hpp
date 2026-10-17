// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/qsa_indexer.hpp"

namespace ov::intel_gpu::op {

/// @brief GPU-internal non-paged (caller-owned dense cache) QSA Indexer. Reuses
/// ov::op::internal::QSAIndexer's Config/InputIdx/shape-inference/attribute-visiting
/// logic wholesale; the only GPU-specific bit is the "gpu_opset" RTTI tag.
class QSAIndexer : public ov::op::internal::QSAIndexer {
public:
    OPENVINO_OP("QSAIndexer", "gpu_opset", ov::op::internal::QSAIndexer);

    using Base = ov::op::internal::QSAIndexer;
    using Config = Base::Config;
    using InputIdx = Base::InputIdx;
    using Base::validate_config;

    QSAIndexer() = default;

    QSAIndexer(const ov::OutputVector& args, const Config& config) : Base(args, config) {}

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace ov::intel_gpu::op
