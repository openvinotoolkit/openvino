// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/swiglu.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/swiglu.hpp"

namespace ov {
namespace op {
namespace internal {
using SwiGLU = ov::intel_gpu::op::SwiGLU;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov {
namespace intel_gpu {

static void CreateSwiGLUOp(ProgramBuilder& p, const std::shared_ptr<op::SwiGLU>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string primitive_name = layer_type_name_ID(op);

    if (p.use_new_shape_infer()) {
        auto prim = cldnn::swiglu(primitive_name,
                                  inputs[0],
                                  op->get_axis(),
                                  op->get_split_lengths(),
                                  op->get_glu_type(),
                                  op->get_split_to_glu_idx(),
                                  cldnn::tensor());
        prim.output_data_types = get_output_data_types(op);
        p.add_primitive(*op, prim);
    } else {
        auto prim = cldnn::swiglu(primitive_name,
                                  inputs[0],
                                  op->get_axis(),
                                  op->get_split_lengths(),
                                  op->get_glu_type(),
                                  op->get_split_to_glu_idx(),
                                  tensor_from_dims(op->get_output_shape(0)));
        prim.output_data_types = get_output_data_types(op);
        p.add_primitive(*op, prim);
    }
}

REGISTER_FACTORY_IMPL(internal, SwiGLU);

}  // namespace intel_gpu
}  // namespace ov
