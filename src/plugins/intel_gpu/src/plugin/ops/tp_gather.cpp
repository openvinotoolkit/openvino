// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_TP

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/tp_gather.hpp"

#include "ov_ops/tp_collective.hpp"

namespace ov::intel_gpu {

static void CreateTPGatherOp(ProgramBuilder& p,
                             const std::shared_ptr<ov::op::internal::TPGather>& op) {
    validate_inputs_count(op, {1});
    OPENVINO_ASSERT(op->get_group_id() == 0,
                    "TPGather ", op->get_friendly_name(), ": only collective group 0 is supported, got ", op->get_group_id());
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    // Counted from the front: static layouts are padded to 4D at the back, where -1 would land on a padding axis.
    const auto input_rank = op->get_input_partial_shape(0).rank();
    OPENVINO_ASSERT(input_rank.is_static(), "TPGather ", op->get_friendly_name(), ": the input rank must be static");
    const int64_t axis = op->get_axis() < 0 ? op->get_axis() + input_rank.get_length() : op->get_axis();

    auto prim = cldnn::tp_gather(layerName,
                                 inputs[0],
                                 p.next_tp_collective_ordinal(),
                                 static_cast<uint32_t>(op->get_rank()),
                                 static_cast<uint32_t>(op->get_world_size()),
                                 axis);

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, TPGather);

}  // namespace ov::intel_gpu

#endif  // ENABLE_TP
