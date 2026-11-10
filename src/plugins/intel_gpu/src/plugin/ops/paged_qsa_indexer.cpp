// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/paged_qsa_indexer.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/paged_qsa_indexer.hpp"

namespace ov::op::internal {
// The core ov::op::internal::PagedQSAIndexer already owns the plain name; like MLIR/MLIROp, the alias differs.
using PagedQSAIndexerGPU = ov::intel_gpu::op::PagedQSAIndexer;
}  // namespace ov::op::internal

namespace ov::intel_gpu {

static void CreatePagedQSAIndexerGPUOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::PagedQSAIndexerGPU>& op) {
    using In = ov::intel_gpu::op::PagedQSAIndexer::InputIdx;
    validate_inputs_count(op, {In::COUNT});
    const auto inputs = p.GetInputInfo(op);
    const auto layer_name = layer_type_name_ID(op);

    std::string reason;
    OPENVINO_ASSERT(ov::intel_gpu::op::PagedQSAIndexer::validate_config(op->get_config(), reason),
                    "[GPU] Cannot lower PagedQSAIndexer op '",
                    layer_name,
                    "': ",
                    reason);

    OPENVINO_ASSERT(op->get_config().num_selection_heads == 1,
                    "[GPU] PagedQSAIndexer '",
                    layer_name,
                    "': only num_selection_heads == 1 is supported, got ",
                    op->get_config().num_selection_heads);
    OPENVINO_ASSERT(op->get_index_key_heads() == 1, "[GPU] PagedQSAIndexer '", layer_name, "': only H_ik == 1 is supported, got ", op->get_index_key_heads());
    for (size_t idx : {In::INDEX_QUERY,
                       In::RAW_INDEX_KEYS,
                       In::ROPE_COS,
                       In::ROPE_SIN,
                       In::BLOCK_INDEX_KEYS_CACHE,
                       In::INDEX_KEY_CACHE,
                       In::Q_NORM_WEIGHT,
                       In::K_NORM_WEIGHT,
                       In::INDEX_BLOCK_ROPE_CACHE}) {
        OPENVINO_ASSERT(op->get_input_element_type(idx) == ov::element::f16,
                        "[GPU] PagedQSAIndexer '",
                        layer_name,
                        "': input ",
                        idx,
                        " must be f16, got ",
                        op->get_input_element_type(idx));
    }

    cldnn::paged_qsa_indexer prim(layer_name, inputs, op->get_config());
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, PagedQSAIndexerGPU);

}  // namespace ov::intel_gpu
