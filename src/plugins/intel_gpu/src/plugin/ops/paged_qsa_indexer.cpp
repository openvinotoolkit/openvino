// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_qsa_indexer.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/paged_qsa_indexer.hpp"

// NOTE: like ops/qsa.cpp, PagedQSAIndexer is lowered from the base
// ov::op::internal::PagedQSAIndexer type; ov::intel_gpu::op::PagedQSAIndexer
// ("gpu_opset") publicly inherits from it and adds no new fields, so
// ProgramBuilder::CreateSingleLayerPrimitive's type_info->parent walk already dispatches
// gpu_opset nodes to a factory registered for the base type.

namespace ov::intel_gpu {

static void CreatePagedQSAIndexerOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::PagedQSAIndexer>& op) {
    validate_inputs_count(op, {ov::op::internal::PagedQSAIndexer::InputIdx::COUNT});
    const auto inputs = p.GetInputInfo(op);
    const auto layer_name = layer_type_name_ID(op);

    std::string reason;
    OPENVINO_ASSERT(ov::op::internal::PagedQSAIndexer::validate_config(op->get_config(), reason),
                    "[GPU] Cannot lower PagedQSAIndexer op '",
                    layer_name,
                    "': ",
                    reason);

    cldnn::paged_qsa_indexer prim(layer_name, inputs, op->get_config());
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, PagedQSAIndexer);

}  // namespace ov::intel_gpu
