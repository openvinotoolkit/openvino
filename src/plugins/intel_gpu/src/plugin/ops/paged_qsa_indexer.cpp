// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_qsa_indexer.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/paged_qsa_indexer.hpp"

namespace ov::intel_gpu {

static void CreatePagedQSAIndexerOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::PagedQSAIndexer>& op) {
    using In = ov::op::internal::PagedQSAIndexer::InputIdx;
    validate_inputs_count(op, {In::COUNT});
    const auto inputs = p.GetInputInfo(op);
    const auto layer_name = layer_type_name_ID(op);

    std::string reason;
    OPENVINO_ASSERT(ov::op::internal::PagedQSAIndexer::validate_config(op->get_config(), reason),
                    "[GPU] Cannot lower PagedQSAIndexer op '",
                    layer_name,
                    "': ",
                    reason);

    OPENVINO_ASSERT(op->get_config().query_norm && op->get_config().query_rope,
                    "[GPU] PagedQSAIndexer '",
                    layer_name,
                    "': only query_norm == true and query_rope == true are supported");
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

REGISTER_FACTORY_IMPL(internal, PagedQSAIndexer);

}  // namespace ov::intel_gpu
