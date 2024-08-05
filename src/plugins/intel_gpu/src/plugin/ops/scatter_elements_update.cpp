// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/scatter_elements_update.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/scatter_elements_update.hpp"

namespace ov {
namespace intel_gpu {

static void CreateScatterElementsUpdateOp(ProgramBuilder& p, const std::shared_ptr<op::util::ScatterElementsUpdateBase>& op) {
    validate_inputs_count(op, {4});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto axes_constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(3));
    OPENVINO_ASSERT(axes_constant, "Unsupported parameter nodes type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");

    int64_t axis = ov::util::try_normalize_axis(axes_constant->cast_vector<int64_t>()[0],
                                                op->get_input_partial_shape(0).rank(),
                                                *op);

    auto mode = cldnn::ScatterElementsUpdateOp::Reduction::NONE;
    auto use_init_val = true;
    if (const auto op_v12 = std::dynamic_pointer_cast<cldnn::ScatterElementsUpdateOp>(op)) {
        mode = op_v12->get_reduction();
        use_init_val = op_v12->get_use_init_val();
    }

    auto primitive = cldnn::scatter_elements_update(layerName,
                                                    inputs[0],
                                                    inputs[1],
                                                    inputs[2],
                                                    axis,
                                                    mode,
                                                    use_init_val);

    p.add_primitive(*op, primitive);
}

REGISTER_FACTORY_IMPL(v3, ScatterElementsUpdate);
REGISTER_FACTORY_IMPL(v12, ScatterElementsUpdate);

}  // namespace intel_gpu
}  // namespace ov
