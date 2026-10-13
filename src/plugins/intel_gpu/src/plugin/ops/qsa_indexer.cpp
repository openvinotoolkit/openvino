// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/qsa_indexer.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/qsa_indexer.hpp"

namespace ov::intel_gpu {

static void CreateQSAIndexerOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::QSAIndexer>& op) {
    validate_inputs_count(op, {ov::op::internal::QSAIndexer::InputIdx::COUNT});
    const auto inputs = p.GetInputInfo(op);
    const auto layer_name = layer_type_name_ID(op);

    std::string reason;
    OPENVINO_ASSERT(ov::op::internal::QSAIndexer::validate_config(op->get_config(), reason),
                    "[GPU] Cannot lower QSAIndexer op '",
                    layer_name,
                    "': ",
                    reason);

    cldnn::qsa_indexer prim(layer_name, inputs, op->get_config());
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, QSAIndexer);

}  // namespace ov::intel_gpu
