// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "openvino/op/random_uniform.hpp"
#include "intel_gpu/primitives/random_uniform.hpp"


namespace ov::intel_gpu {

namespace {

void CreateRandomUniformOp(ProgramBuilder &p, const std::shared_ptr<ov::op::v8::RandomUniform> &op) {
    auto inputs = p.GetInputInfo(op);
    auto input_pshape = op->get_input_partial_shape(0);
    auto output_pshape = op->get_output_partial_shape(0);

    OPENVINO_ASSERT(input_pshape.is_static(), "[GPU] Dynamic input of RandomUniform leads to dynamic output rank, but GPU doesn't support it yet");

    if (output_pshape.is_static() && !p.use_new_shape_infer()) {
        auto output_shape = output_pshape.get_shape();
        // Extend to 4D shape
        output_shape.insert(output_shape.end(), 4 - output_shape.size(), 1ul);

        auto random_uniform_prim = cldnn::random_uniform(layer_type_name_ID(op),
                                                         inputs,
                                                         cldnn::element_type_to_data_type(op->get_out_type()),
                                                         op->get_global_seed(),
                                                         op->get_op_seed(),
                                                         output_shape);
        p.add_primitive(*op, random_uniform_prim);
    } else {
        OPENVINO_ASSERT(input_pshape.size() == 1, "[GPU] RandomUniform expects 1D input, got ", input_pshape.size());

        auto random_uniform_prim = cldnn::random_uniform(layer_type_name_ID(op),
                                                         inputs,
                                                         cldnn::element_type_to_data_type(op->get_out_type()),
                                                         op->get_global_seed(),
                                                         op->get_op_seed());
        p.add_primitive(*op, random_uniform_prim);
    }
}

} // namespace

REGISTER_FACTORY_IMPL(v8, RandomUniform);

}  // namespace ov::intel_gpu
