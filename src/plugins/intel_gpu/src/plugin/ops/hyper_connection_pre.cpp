// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/hyper_connection_pre.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/hyper_connection_pre.hpp"

namespace ov::intel_gpu {

static void CreateHyperConnectionPreOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::HyperConnectionPre>& op) {
    validate_inputs_count(op, {4});
    // The GPU kernel maps the n x n Sinkhorn matrix onto one 16-lane subgroup, so it covers n == 4 only.
    OPENVINO_ASSERT(op->get_n_streams() == 4,
                    "GPU HyperConnectionPre supports n_streams == 4 only, got ",
                    op->get_n_streams(),
                    " for ",
                    op->get_friendly_name());
    auto inputs = p.GetInputInfo(op);
    cldnn::hyper_connection_pre prim(layer_type_name_ID(op),
                                     inputs,
                                     op->get_n_streams(),
                                     op->get_sinkhorn_iterations(),
                                     op->get_eps(),
                                     op->get_rms_eps());
    prim.num_outputs = op->get_output_size();
    prim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, HyperConnectionPre);

}  // namespace ov::intel_gpu
