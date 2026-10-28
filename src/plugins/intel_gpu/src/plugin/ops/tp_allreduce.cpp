// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_TP

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/tp_allreduce.hpp"

#include "ov_ops/tp_collective.hpp"

namespace ov::intel_gpu {

static void CreateTPAllReduceOp(ProgramBuilder& p,
                                const std::shared_ptr<ov::op::internal::TPAllReduce>& op) {
    validate_inputs_count(op, {1});
    OPENVINO_ASSERT(op->get_reduce_op() == ov::op::internal::ReduceOp::SUM,
                    "TPAllReduce ", op->get_friendly_name(), ": only SUM is supported");
    OPENVINO_ASSERT(op->get_group_id() == 0,
                    "TPAllReduce ", op->get_friendly_name(), ": only collective group 0 is supported, got ", op->get_group_id());
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto prim = cldnn::tp_allreduce(layerName,
                                    inputs[0],
                                    p.next_tp_collective_ordinal(),
                                    static_cast<uint32_t>(op->get_rank()),
                                    static_cast<uint32_t>(op->get_world_size()));

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, TPAllReduce);

}  // namespace ov::intel_gpu

#endif  // ENABLE_TP
