// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/swoosh.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/activation.hpp"

namespace ov::op::internal {
using Swoosh = ov::intel_gpu::op::Swoosh;
}

namespace ov::intel_gpu {

static void CreateSwooshOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::Swoosh>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string primitive_name = layer_type_name_ID(op);
    auto primitive = cldnn::activation(primitive_name,
                                       inputs[0],
                                       cldnn::activation_func::swoosh,
                                       {op->get_offset(), op->get_bias()});
    p.add_primitive(*op, primitive);
}

REGISTER_FACTORY_IMPL(internal, Swoosh);

}  // namespace ov::intel_gpu
