// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_TP_GPU

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/tp_gather.hpp"

#include "tp_gpu/op/tp_gather.hpp"

namespace ov {
namespace op {
namespace tp_gpu {
using TPGather = ov::tp_gpu::op::TPGather;
}  // namespace tp_gpu
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

static void CreateTPGatherOp(ProgramBuilder& p,
                             const std::shared_ptr<ov::tp_gpu::op::TPGather>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto prim = cldnn::tp_gather(layerName,
                                 inputs[0],
                                 op->get_group_id(),
                                 op->get_collective_id(),
                                 op->get_rank(),
                                 op->get_world_size(),
                                 op->get_axis());

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(tp_gpu, TPGather);

}  // namespace ov::intel_gpu

#endif  // ENABLE_TP_GPU
