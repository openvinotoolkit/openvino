// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "transformations/utils/utils.hpp"

#include "openvino/op/one_hot.hpp"
#include "intel_gpu/primitives/one_hot.hpp"

namespace ov::intel_gpu {

static void CreateOneHotOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::OneHot>& op) {
    validate_inputs_count(op, {4});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    int64_t axis = op->get_axis();
    auto depth_value_node = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    auto on_value_node = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(2));
    auto off_value_node = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(3));

    OPENVINO_ASSERT(on_value_node != nullptr || off_value_node != nullptr || depth_value_node != nullptr,
                    "[GPU] Unsupported on/off/depth nodes type in ", op->get_friendly_name(), " (", op->get_type_name(), ")");

    float on_value;
    float off_value;

    if (!ov::op::util::get_single_value(on_value_node, on_value) ||
        !ov::op::util::get_single_value(off_value_node, off_value)) {
        OPENVINO_THROW("Unsupported parameter size in ", op->get_friendly_name(), " (", op->get_type_name(), ")");
    }

    auto dims = op->get_input_partial_shape(0);

    if (axis < -1 || axis > static_cast<int16_t>(dims.size()))
        OPENVINO_THROW(op->get_friendly_name(), " Incorrect OneHot axis value: ", axis, ". Should be between -1 and ", dims.size());

    if (axis == -1) {
        axis = dims.size();
        for (int i = static_cast<int>(dims.size() - 1); i >= 0; i--) {
            if (dims[i] == 1)
                axis--;
            else
                break;
        }
    }

    auto out_pshape = op->get_output_partial_shape(0);
    cldnn::tensor out_tensor = out_pshape.is_static() ? tensor_from_dims(out_pshape.to_shape()) : cldnn::tensor{};

    if (depth_value_node) {
        int64_t depth = depth_value_node->cast_vector<int64_t>()[0];
        auto oneHotPrim = cldnn::one_hot(layerName,
                                         inputs[0],
                                         out_tensor,
                                         cldnn::element_type_to_data_type(op->get_output_element_type(0)),
                                         axis,
                                         depth,
                                         on_value,
                                         off_value);

        p.add_primitive(*op, oneHotPrim);
    } else {
        auto oneHotPrim = cldnn::one_hot(layerName,
                                         inputs[0],
                                         inputs[1],
                                         out_tensor,
                                         cldnn::element_type_to_data_type(op->get_output_element_type(0)),
                                         axis,
                                         on_value,
                                         off_value);

        p.add_primitive(*op, oneHotPrim);
    }
}

REGISTER_FACTORY_IMPL(v1, OneHot);

}  // namespace ov::intel_gpu
