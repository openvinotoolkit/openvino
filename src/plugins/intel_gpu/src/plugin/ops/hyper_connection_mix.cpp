// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/hyper_connection_mix.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/hyper_connection_mix.hpp"

namespace ov::intel_gpu {

static void CreateHyperConnectionMixOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::HyperConnectionMix>& op) {
    validate_inputs_count(op, {3});
    auto inputs = p.GetInputInfo(op);
    cldnn::hyper_connection_mix prim(layer_type_name_ID(op), inputs, op->get_n_streams(), op->get_sinkhorn_iterations(), op->get_eps());
    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, HyperConnectionMix);

}  // namespace ov::intel_gpu
