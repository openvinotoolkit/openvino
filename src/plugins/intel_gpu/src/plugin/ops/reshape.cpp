// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/validation_util.hpp"

#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/reorder.hpp"

namespace ov::intel_gpu {

static void CreateCommonReshapeOp(ProgramBuilder& p, const std::shared_ptr<ov::Node>& op, cldnn::reshape::reshape_mode mode, bool special_zero = false) {
    validate_inputs_count(op, {1, 2});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto input_pshape = op->get_input_partial_shape(0);
    auto output_pshape = op->get_output_partial_shape(0);

    if (p.use_new_shape_infer() || op->is_dynamic()) {
        std::shared_ptr<cldnn::reshape> reshape_prim = nullptr;
        auto second_const_input = op->get_input_size() == 2 ? ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1)) : nullptr;
        std::vector<int64_t> output_pattern = {};
        if (second_const_input != nullptr) {
            output_pattern = second_const_input->cast_vector<int64_t>();
            if (mode == cldnn::reshape::reshape_mode::unsqueeze) {
                ov::util::try_normalize_axes(output_pattern, op->get_output_partial_shape(0).rank(), *op);
            } else if (mode == cldnn::reshape::reshape_mode::squeeze) {
                ov::util::try_normalize_axes(output_pattern, op->get_input_partial_shape(0).rank(), *op);
            }
        }

        // If second input is absent (it's optional in Squeeze op) or it's constant, create reshape with single input and compile time out pattern
        if (op->get_input_size() == 1 || second_const_input != nullptr) {
            reshape_prim = std::make_shared<cldnn::reshape>(layerName,
                                                            inputs[0],
                                                            special_zero,
                                                            output_pattern,
                                                            output_pshape,
                                                            mode);
        } else {
            reshape_prim = std::make_shared<cldnn::reshape>(layerName,
                                                            inputs[0],
                                                            inputs[1],
                                                            special_zero,
                                                            output_pshape,
                                                            mode);
        }

        p.add_primitive(*op, reshape_prim);
    } else {
        OPENVINO_ASSERT(input_pshape.is_static() && output_pshape.is_static(), "Dynamic shapes are not supported for Reshape operation yet");

        auto outTensor = tensor_from_dims(output_pshape.to_shape());

        // if we convert from or to 5D/6D, additional reorder also required to change format
        cldnn::input_info reshape_input = inputs[0];
        if (input_pshape.size() != output_pshape.size()) {
            cldnn::primitive_id reorderId = "reorder:" + op->get_friendly_name() + "_reorder";
            cldnn::format outputFormat = cldnn::format::bfyx;

            switch (output_pshape.size()) {
            case 5: outputFormat = cldnn::format::bfzyx; break;
            case 6: outputFormat = cldnn::format::bfwzyx; break;
            default: break;
            }

            cldnn::layout outputLayout(cldnn::element_type_to_data_type(op->get_output_element_type(0)), outputFormat, outTensor);
            p.add_primitive(*op, cldnn::reorder(reorderId,
                                                reshape_input,
                                                outputLayout));
            reshape_input = cldnn::input_info(reorderId);
        }

        auto reshapePrim = cldnn::reshape(layerName,
                                        reshape_input,
                                        outTensor,
                                        mode);

        p.add_primitive(*op, reshapePrim);
    }
}

static void CreateReshapeOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::Reshape>& op) {
    CreateCommonReshapeOp(p, op, cldnn::reshape::reshape_mode::base, op->get_special_zero());
}

static void CreateSqueezeOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Squeeze>& op) {
    CreateCommonReshapeOp(p, op, cldnn::reshape::reshape_mode::squeeze);
}

static void CreateUnsqueezeOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Unsqueeze>& op) {
    CreateCommonReshapeOp(p, op, cldnn::reshape::reshape_mode::unsqueeze);
}

REGISTER_FACTORY_IMPL(v1, Reshape);
REGISTER_FACTORY_IMPL(v0, Squeeze);
REGISTER_FACTORY_IMPL(v0, Unsqueeze);

}  // namespace ov::intel_gpu
