// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"

#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/validation_util.hpp"

#include "intel_gpu/primitives/reshape.hpp"

namespace ov {
namespace intel_gpu {

static void CreateCommonReshapeOp(ProgramBuilder& p, const std::shared_ptr<ov::Node>& op, cldnn::reshape::reshape_mode mode, bool special_zero = false) {
    validate_inputs_count(op, {1, 2});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto input_pshape = op->get_input_partial_shape(0);
    auto output_pshape = op->get_output_partial_shape(0);

    std::shared_ptr<cldnn::reshape> reshape_prim = nullptr;
    auto second_const_input = op->get_input_size() == 2 ? std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1)) : nullptr;
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

}  // namespace intel_gpu
}  // namespace ov
