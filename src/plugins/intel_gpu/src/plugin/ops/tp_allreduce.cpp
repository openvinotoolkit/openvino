// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_TP

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/tp_allreduce.hpp"

#include "tp/op/tp_all_reduce.hpp"

namespace ov {
namespace op {
namespace tp {
using TPAllReduce = ov::tp::op::TPAllReduce;
}  // namespace tp
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

static void CreateTPAllReduceOp(ProgramBuilder& p,
                                const std::shared_ptr<ov::tp::op::TPAllReduce>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto prim = cldnn::tp_allreduce(layerName,
                                    inputs[0],
                                    op->get_collective_id(),
                                    op->get_rank());

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(tp, TPAllReduce);

}  // namespace ov::intel_gpu

#endif  // ENABLE_TP
