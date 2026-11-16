// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/paged_qsa_indexer.hpp"

namespace ov::intel_gpu::op {

/// @brief GPU-internal paged-cache QSA Indexer. Reuses ov::op::internal::PagedQSAIndexer's
/// Config/InputIdx/shape-inference/attribute-visiting logic wholesale; the only
/// GPU-specific bit is the "gpu_opset" RTTI tag used by ProgramBuilder dispatch and by
/// the cldnn::paged_qsa_indexer primitive lowering.
class PagedQSAIndexer : public ov::op::internal::PagedQSAIndexer {
public:
    OPENVINO_OP("PagedQSAIndexer", "gpu_opset", ov::op::internal::PagedQSAIndexer);

    using Base = ov::op::internal::PagedQSAIndexer;
    using Config = Base::Config;
    using InputIdx = Base::InputIdx;
    using Base::validate_config;

    PagedQSAIndexer() = default;

    PagedQSAIndexer(const ov::OutputVector& args, const Config& config) : Base(args, config) {}

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace ov::intel_gpu::op
