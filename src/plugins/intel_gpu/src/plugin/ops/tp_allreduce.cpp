// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_TENSOR_PARALLEL

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/tp_allreduce.hpp"

#include "tensor_parallel/tp_coordination.hpp"
#include "tensor_parallel/op/tp_all_reduce.hpp"

namespace ov::intel_gpu {

static void CreateTPAllReduceOp(ProgramBuilder& p,
                                const std::shared_ptr<ov::op::tp::TPAllReduce>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    // Read coordination context and rank from rt_info (stored by graph_rewriter).
    const auto& rt = op->get_rt_info();

    auto coord_it = rt.find("tp_coordination");
    OPENVINO_ASSERT(coord_it != rt.end(),
                    "[GPU] TPAllReduce '", op->get_friendly_name(),
                    "' is missing tp_coordination in rt_info");
    auto coordination = coord_it->second.as<std::shared_ptr<ov::tp::TPCoordination>>();

    auto rank_it = rt.find("tp_rank");
    OPENVINO_ASSERT(rank_it != rt.end(),
                    "[GPU] TPAllReduce '", op->get_friendly_name(),
                    "' is missing tp_rank in rt_info");
    auto rank = static_cast<uint32_t>(rank_it->second.as<int64_t>());

    auto prim = cldnn::tp_allreduce(layerName,
                                    inputs[0],
                                    op->get_collective_id(),
                                    rank,
                                    std::move(coordination));

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(tp, TPAllReduce);

}  // namespace ov::intel_gpu

#endif  // ENABLE_TENSOR_PARALLEL
