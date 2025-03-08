// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/dynamic_quantize.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/dynamic_quantize.hpp"


namespace ov::intel_gpu {

static void CreateDynamicQuantizeOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::DynamicQuantize>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string primitive_name = layer_type_name_ID(op);

    auto prim = cldnn::dynamic_quantize(primitive_name,
                                        inputs[0],
                                        op->get_attrs(),
                                        op->get_input_partial_shape(0).size());

    prim.num_outputs = op->get_output_size();

    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(internal, DynamicQuantize);

}  // namespace ov::intel_gpu
