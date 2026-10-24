// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/qsa_indexer.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/qsa_indexer.hpp"

namespace ov::op::internal {
using QSAIndexer = ov::intel_gpu::op::QSAIndexer;
}  // namespace ov::op::internal

namespace ov::intel_gpu {

static void CreateQSAIndexerOp(ProgramBuilder& p, const std::shared_ptr<ov::intel_gpu::op::QSAIndexer>& op) {
    validate_inputs_count(op, {ov::intel_gpu::op::QSAIndexer::InputIdx::COUNT});
    const auto inputs = p.GetInputInfo(op);
    const auto layer_name = layer_type_name_ID(op);

    std::string reason;
    OPENVINO_ASSERT(ov::intel_gpu::op::QSAIndexer::validate_config(op->get_config(), reason),
                    "[GPU] Cannot lower QSAIndexer op '",
                    layer_name,
                    "': ",
                    reason);

    using In = ov::intel_gpu::op::QSAIndexer::InputIdx;
    OPENVINO_ASSERT(op->get_config().num_selection_heads == 1, "[GPU] QSAIndexer '", layer_name, "': only num_selection_heads == 1 is supported, got ", op->get_config().num_selection_heads);
    OPENVINO_ASSERT(op->get_index_key_heads() == 1, "[GPU] QSAIndexer '", layer_name, "': only H_ik == 1 is supported, got ", op->get_index_key_heads());
    for (size_t idx : {In::INDEX_QUERY,
                       In::RAW_INDEX_KEYS,
                       In::ROPE_COS,
                       In::ROPE_SIN,
                       In::Q_NORM_WEIGHT,
                       In::K_NORM_WEIGHT,
                       In::BLOCK_INDEX_KEYS_CACHE,
                       In::PAST_INCOMPLETE_RAW_K,
                       In::PAST_INCOMPLETE_ROPE}) {
        OPENVINO_ASSERT(op->get_input_element_type(idx) == ov::element::f16,
                        "[GPU] QSAIndexer '", layer_name, "': input ", idx, " must be f16, got ", op->get_input_element_type(idx));
    }

    cldnn::qsa_indexer prim(layer_name, inputs, op->get_config());
    prim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, QSAIndexer);

}  // namespace ov::intel_gpu
