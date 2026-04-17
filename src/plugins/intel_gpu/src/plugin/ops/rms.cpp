// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/rms.hpp"
#include <ov_ops/rms.hpp>

using RMS = ov::op::internal::RMS;

namespace ov::intel_gpu {

static void CreateRMSOp(ProgramBuilder& p, const std::shared_ptr<RMS>& op) {
    validate_inputs_count(op, {1, 2});
    auto inputs = p.GetInputInfo(op);
    std::string primitive_name = layer_type_name_ID(op);

    auto rms = op->get_elementwise_affine()
        ? cldnn::rms(primitive_name, inputs[0], inputs[1], op->get_epsilon())
        : cldnn::rms(primitive_name, inputs[0], op->get_epsilon());
    rms.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, rms);
}

REGISTER_FACTORY_IMPL(internal, RMS);

}  // namespace ov::intel_gpu
