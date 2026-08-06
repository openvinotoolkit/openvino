// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_TP_GPU

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/tp_allreduce.hpp"

#include "tp_gpu/op/tp_all_reduce.hpp"

namespace ov {
namespace op {
namespace tp_gpu {
using TPAllReduce = ov::tp_gpu::op::TPAllReduce;
}  // namespace tp_gpu
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

static void CreateTPAllReduceOp(ProgramBuilder& p,
                                const std::shared_ptr<ov::tp_gpu::op::TPAllReduce>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    // The op carries all the metadata the primitive needs; the coordinator that
    // runs the group is resolved at execution time from the network's registry.
    auto prim = cldnn::tp_allreduce(layerName,
                                    inputs[0],
                                    op->get_group_id(),
                                    op->get_collective_id(),
                                    op->get_rank());

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(tp_gpu, TPAllReduce);

}  // namespace ov::intel_gpu

#endif  // ENABLE_TP_GPU
