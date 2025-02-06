// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/lstm_sequence.hpp"

#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/lstm_cell.hpp"
#include "intel_gpu/primitives/crop.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/permute.hpp"
#include "intel_gpu/primitives/slice.hpp"

namespace ov::intel_gpu {
static cldnn::activation_func GetActivationFunc(std::string name) {
    static const std::map<std::string, cldnn::activation_func> name_mapping = {
        {"sigmoid", cldnn::activation_func::logistic},
        {"tanh", cldnn::activation_func::hyperbolic_tan},
        {"relu", cldnn::activation_func::relu},
    };
    auto itr = name_mapping.find(name);
    if (itr != name_mapping.end())
        return itr->second;
    else
        return cldnn::activation_func::none;
}

template <typename T>
void GetLSTMActivationParams(const std::shared_ptr<T>& op,
                             std::vector<cldnn::activation_func>& activations,
                             std::vector<cldnn::activation_additional_params>& activation_params) {
    activations = { cldnn::activation_func::logistic,
                    cldnn::activation_func::hyperbolic_tan,
                    cldnn::activation_func::hyperbolic_tan };
    activation_params = {};
    auto op_activations = op->get_activations();
    if (!op_activations.empty()) {
        if (op_activations.size() != 3)
            OPENVINO_THROW("Wrong number of activations for LSTMCell op ", op->get_friendly_name());
        for (int i = 0; i < 3; i++) {
            auto af = GetActivationFunc(op_activations[i]);
            if (af == cldnn::activation_func::none)
                OPENVINO_THROW("Wrong or unsupported activation type ", op_activations[i], " for LSTMCell op ", op->get_friendly_name());
            activations[i] = af;
        }
    }
    auto op_a = op->get_activations_alpha();
    auto op_b = op->get_activations_beta();
    if (!op_a.empty()) {
        if (op_a.size() != 3 || op_b.size() != 3)
            OPENVINO_THROW("Wrong number of activation parameters for LSTMCell op ", op->get_friendly_name());
        for (int i = 0; i < 3; i++) {
            cldnn::activation_additional_params params = { op_a[i], op_b[i] };
            activation_params.push_back(cldnn::activation_additional_params(params));
        }
    }
}

static void CreateLSTMCellOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v4::LSTMCell>& op) {
    validate_inputs_count(op, {6});
    std::string layerName = layer_type_name_ID(op);
    auto inputs = p.GetInputInfo(op);
    std::vector<cldnn::activation_func> activations;
    std::vector<cldnn::activation_additional_params> activation_params;
    GetLSTMActivationParams(op, activations, activation_params);
    float clip = op->get_clip();
    OPENVINO_ASSERT(!inputs[5].pid.empty());
    OPENVINO_ASSERT(p.use_new_shape_infer());
    p.add_primitive(*op, cldnn::lstm_cell(layerName, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], cldnn::input_info(),
        clip, false, activations, activation_params, cldnn::lstm_weights_order::fizo, ov::op::RecurrentSequenceDirection::FORWARD,
        static_cast<int>(op->get_output_size())));
}

static void CreateLSTMSequenceOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v5::LSTMSequence>& op) {
    validate_inputs_count(op, {7});
    std::string layerName = layer_type_name_ID(op);
    auto inputs = p.GetInputInfo(op);
    std::vector<cldnn::activation_func> activations;
    std::vector<cldnn::activation_additional_params> activation_params;
    GetLSTMActivationParams(op, activations, activation_params);
    const float clip = op->get_clip();
    OPENVINO_ASSERT(op->get_input_shape(2).size() == 3 && op->get_input_shape(3).size() == 1 && op->get_input_shape(4).size() == 3 &&
        op->get_input_shape(5).size() == 3 && op->get_input_shape(6).size() == 2, "Wrong input shapes for LSTMSequence op ", op->get_friendly_name());
    auto direction = op->get_direction();

    OPENVINO_ASSERT(p.use_new_shape_infer());
    cldnn::lstm_seq prim(layerName, inputs[0], inputs[1], inputs[2], inputs[4], inputs[5], inputs[6], inputs[3], clip, false, activations,
        activation_params, cldnn::lstm_weights_order::fizo, direction, static_cast<int>(op->get_output_size()));
    prim.output_data_types = get_output_data_types(op);
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(v4, LSTMCell);
REGISTER_FACTORY_IMPL(v5, LSTMSequence);

}  // namespace ov::intel_gpu
