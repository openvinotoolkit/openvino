// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/gated_mlp.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/gated_mlp.hpp"

namespace ov {
namespace op {
namespace internal {
using GatedMLP = ov::intel_gpu::op::GatedMLP;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {

static void CreateGatedMLPOp(ProgramBuilder& p, const std::shared_ptr<ov::intel_gpu::op::GatedMLP>& op) {
    validate_inputs_count(op, {4});
    auto inputs = p.GetInputInfo(op);
    const auto layer_name = layer_type_name_ID(op);

    if (p.use_new_shape_infer()) {
        auto prim = cldnn::gated_mlp(layer_name,
                                     cldnn::input_info(inputs[0]),
                                     cldnn::input_info(inputs[1]),
                                     cldnn::input_info(inputs[2]),
                                     cldnn::input_info(inputs[3]),
                                     op->get_activation(),
                                     cldnn::tensor(),
                                     cldnn::element_type_to_data_type(op->get_output_element_type(0)));
        prim.output_data_types = get_output_data_types(op);
        p.add_primitive(*op, prim);
    } else {
        auto output_pshape = op->get_output_partial_shape(0);
        OPENVINO_ASSERT(output_pshape.is_static(), "GatedMLP requires static output shape at primitive creation.");
        auto prim = cldnn::gated_mlp(layer_name,
                                     cldnn::input_info(inputs[0]),
                                     cldnn::input_info(inputs[1]),
                                     cldnn::input_info(inputs[2]),
                                     cldnn::input_info(inputs[3]),
                                     op->get_activation(),
                                     tensor_from_dims(output_pshape.to_shape()),
                                     cldnn::element_type_to_data_type(op->get_output_element_type(0)));
        p.add_primitive(*op, prim);
    }
}

REGISTER_FACTORY_IMPL(internal, GatedMLP);

}  // namespace ov::intel_gpu
