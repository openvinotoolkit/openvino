// Copyright (C) 2018-2024 Intel Corporation
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
#include "intel_gpu/primitives/lstm.hpp"
#include "intel_gpu/primitives/lstm_cell.hpp"
#include "intel_gpu/primitives/crop.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/data.hpp"

namespace ov {
namespace intel_gpu {
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
    unsigned int direction = 0;
    assert(!inputs[5].pid.empty());
    cldnn::primitive_id lstm_fc_id = layerName + "_fully_connected";
    p.add_primitive(*op, cldnn::fully_connected(lstm_fc_id, inputs[0], inputs[3].pid, inputs[5].pid, 3));
    if (p.use_new_shape_infer()) {
        auto prim =  cldnn::lstm_cell({layerName+".out0", cldnn::input_info(lstm_fc_id), inputs[1], inputs[2], inputs[4], \
        cldnn::input_info(), "", "", clip, activations, \
        activation_params, cldnn::lstm_weights_order::fizo, direction, cldnn::padding(), \
        static_cast<int>(op->get_output_size())}, 0);
        p.add_primitive(*op, prim);
        return;
    }

    auto mutable_precision_first = op->get_output_element_type(1);
    cldnn::layout outLayout = cldnn::layout(
            cldnn::element_type_to_data_type(mutable_precision_first),
            cldnn::format::get_default_format(op->get_output_shape(1).size()),
            tensor_from_dims(op->get_output_shape(1)));

    cldnn::memory::ptr shared_memory = p.get_engine().allocate_memory(outLayout);
    const cldnn::primitive_id mutable_id_1 = layerName + "_md_write.1";
    const cldnn::mutable_data mutable_prim_1{mutable_id_1, shared_memory};
    p.add_primitive(*op, mutable_prim_1);

    p.add_primitive(*op, cldnn::lstm_cell({layerName+".out0", cldnn::input_info(lstm_fc_id), inputs[1], inputs[2], inputs[4], \
    cldnn::input_info(), layerName + "_md_write.1", "", clip, activations, \
                                        activation_params, cldnn::lstm_weights_order::fizo}, 0));

    p.add_primitive(*op, cldnn::mutable_data(layerName + ".out1", {cldnn::input_info(layerName + ".out0")}, shared_memory));
}

static void CreateLSTMSequenceOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v5::LSTMSequence>& op) {
    validate_inputs_count(op, {7});
    std::string layerName = layer_type_name_ID(op);
    auto inputs = p.GetInputInfo(op);
    if (op->get_input_shape(2).size() != 3 || op->get_input_shape(3).size() != 1 \
        || op->get_input_shape(4).size() != 3 || op->get_input_shape(5).size() != 3 || op->get_input_shape(6).size() != 2)
        OPENVINO_THROW("Wrong input shapes for LSTMSequence op ", op->get_friendly_name());
    std::vector<cldnn::activation_func> activations;
    std::vector<cldnn::activation_additional_params> activation_params;
    GetLSTMActivationParams(op, activations, activation_params);
    float clip = op->get_clip();
    cldnn::primitive_id lstm_seq_id = layerName;
    auto mutable_precision_firstsecond = op->get_output_element_type(1);
    unsigned int direction = op->get_direction() == ov::op::RecurrentSequenceDirection::REVERSE ? 1 : 0;
    cldnn::primitive_id lstm_fc_id = layerName + "_fully_connected";
    p.add_primitive(*op, cldnn::fully_connected(lstm_fc_id, inputs[0], inputs[4].pid, inputs[6].pid, 3));
    if (p.use_new_shape_infer()) {
        cldnn::lstm_seq prim({layerName, lstm_fc_id, inputs[1], \
            inputs[2], inputs[5], inputs[3], "", "", \
            clip, activations, activation_params, cldnn::lstm_weights_order::fizo, direction, cldnn::padding(), \
            static_cast<int>(op->get_output_size())});
        prim.output_data_types = get_output_data_types(op);
        p.add_primitive(*op, prim);
        return;
    }
    cldnn::layout out12Layout = cldnn::layout(
                cldnn::element_type_to_data_type(mutable_precision_firstsecond),
                cldnn::format::get_default_format(op->get_output_shape(1).size()),
                tensor_from_dims(op->get_output_shape(1)));
    std::vector<cldnn::memory::ptr> shared_memories;
    shared_memories.push_back(p.get_engine().allocate_memory(out12Layout));
    const cldnn::primitive_id mutable_id_1 = layerName + "_md_write.1";
    const cldnn::mutable_data mutable_prim_1{mutable_id_1, shared_memories.front()};
    p.add_primitive(*op, mutable_prim_1);
    shared_memories.push_back(p.get_engine().allocate_memory(out12Layout));
    const cldnn::primitive_id mutable_id_2 = layerName + "_md_write.2";
    const cldnn::mutable_data mutable_prim_2{mutable_id_2, shared_memories.back()};
    p.add_primitive(*op, mutable_prim_2);
    p.add_primitive(*op, cldnn::lstm_seq({lstm_seq_id + ".out0", cldnn::input_info(lstm_fc_id), inputs[1], \
        inputs[2], inputs[5], inputs[3], mutable_id_1, mutable_id_2, \
        clip, activations, activation_params, cldnn::lstm_weights_order::fizo, direction}));
    p.add_primitive(*op, cldnn::mutable_data(lstm_seq_id + ".out1", {cldnn::input_info(lstm_seq_id + ".out0")}, shared_memories.front()));
    p.add_primitive(*op, cldnn::mutable_data(lstm_seq_id + ".out2", {cldnn::input_info(lstm_seq_id + ".out0")}, shared_memories.back()));
}

REGISTER_FACTORY_IMPL(v4, LSTMCell);
REGISTER_FACTORY_IMPL(v5, LSTMSequence);

}  // namespace intel_gpu
}  // namespace ov
