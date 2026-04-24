// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/scaled_shifted_clamp_experimental.hpp"
#include "openvino/op/scaled_shifted_clamp_experimental.hpp"

using Op = ov::op::experimental::ScaledShiftedClamp;

namespace ov::intel_gpu {

static void CreateScaledShiftedClampOp(ProgramBuilder& p, const std::shared_ptr<Op>& op) {
    validate_inputs_count(op, {1});
    const auto  inputs        = p.GetInputInfo(op);
    const auto  primitive_name = layer_type_name_ID(op);

    cldnn::scaled_shifted_clamp_experimental prim(primitive_name,
                                                  inputs[0],
                                                  static_cast<float>(op->get_scale()),
                                                  static_cast<float>(op->get_bias()),
                                                  static_cast<float>(op->get_lo()),
                                                  static_cast<float>(op->get_hi()));
    prim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(experimental, ScaledShiftedClamp);

}  // namespace ov::intel_gpu
