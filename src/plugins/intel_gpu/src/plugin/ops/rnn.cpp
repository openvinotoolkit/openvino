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
#include "intel_gpu/primitives/permute.hpp"
#include "intel_gpu/primitives/slice.hpp"

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
    /*
    if (op->get_input_shape(2).size() != 2 || op->get_input_shape(3).size() != 2 \
        || op->get_input_shape(4).size() != 2 || op->get_input_shape(5).size() != 2)
        OPENVINO_THROW("Wrong input shapes for LSTMCell op ", op->get_friendly_name());
    */
    std::vector<cldnn::activation_func> activations;
    std::vector<cldnn::activation_additional_params> activation_params;
    GetLSTMActivationParams(op, activations, activation_params);
    float clip = op->get_clip();
    unsigned int direction = 0;
    assert(!inputs[5].pid.empty());
    if (p.use_new_shape_infer()) {
        auto prim =  cldnn::lstm_cell({layerName+".out0", inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], \
         cldnn::input_info(), "", "", clip, activations, \
                                            activation_params, cldnn::lstm_weights_order::fizo, direction, cldnn::padding(), \
            static_cast<int>(op->get_output_size())}, 0);
        //prim.output_data_types = get_output_data_types(op);
        p.add_primitive(*op, prim);
        return;
    }
    auto mutable_precision_first = op->get_output_element_type(1);
    cldnn::layout outLayout = cldnn::layout(
            cldnn::element_type_to_data_type(mutable_precision_first),
            cldnn::format::get_default_format(op->get_output_shape(1).size()),
            tensor_from_dims(op->get_output_shape(1)));

    cldnn::memory::ptr shared_memory = p.get_engine().allocate_memory(outLayout);
    const cldnn::primitive_id mutable_id_1 = layerName + "_md_write1";
    const cldnn::mutable_data mutable_prim_1{mutable_id_1, shared_memory};
    p.add_primitive(*op, mutable_prim_1);

    p.add_primitive(*op, cldnn::lstm_cell({layerName + ".out0", inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], \
    cldnn::input_info(), layerName + "_md_write1", "", clip, activations, \
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

    if (p.use_new_shape_infer()) {
        cldnn::lstm_seq prim({layerName, inputs[0], inputs[1], \
            inputs[2], inputs[4], inputs[5], inputs[6], inputs[3], "", "", \
            clip, activations, activation_params, cldnn::lstm_weights_order::fizo, direction, cldnn::padding(), \
            static_cast<int>(op->get_output_size())});
        prim.output_data_types = get_output_data_types(op);
        p.add_primitive(*op, prim);
        return;
    }

    cldnn::layout out12Layout = cldnn::layout(
                cldnn::element_type_to_data_type(mutable_precision_firstsecond),
                cldnn::format::bfyx,
                tensor_from_dims(op->get_output_shape(1)));

    std::vector<cldnn::memory::ptr> shared_memories;
    shared_memories.push_back(p.get_engine().allocate_memory(out12Layout));
    const cldnn::primitive_id mutable_id_1 = layerName + "_md_write1";
    const cldnn::mutable_data mutable_prim_1{mutable_id_1, shared_memories.front()};
    p.add_primitive(*op, mutable_prim_1);
    shared_memories.push_back(p.get_engine().allocate_memory(out12Layout));
    const cldnn::primitive_id mutable_id_2 = layerName + "_md_write2";
    const cldnn::mutable_data mutable_prim_2{mutable_id_2, shared_memories.back()};
    p.add_primitive(*op, mutable_prim_2);
    const cldnn::primitive_id permute_unsqueeze_1 = layerName + "_permute_unqsueeze_W";
    const cldnn::primitive_id permute_unsqueeze_R = layerName + "_permute_unqsueeze_R";
    const cldnn::primitive_id permute_id_1 = layerName + "_permute1";
    const cldnn::primitive_id permute_id_2 = layerName + "_permute2";
    const cldnn::primitive_id permute_id_3 = layerName + "_permute3";
    const cldnn::primitive_id permute_W = layerName + "_permuteW";
    const cldnn::primitive_id permute_W2 = layerName + "_permuteW2";
    const cldnn::primitive_id crop_id_W_0 = layerName + "_crop_W_0";
    const cldnn::primitive_id crop_id_W_1 = layerName + "_crop_W_1";
    const cldnn::primitive_id crop_id_W_2 = layerName + "_crop_W_2";
    const cldnn::primitive_id crop_id_W_3 = layerName + "_crop_W_3";
    const cldnn::primitive_id concat_id_W = layerName + "_concat";
    const cldnn::primitive_id crop_id_R_0 = layerName + "_crop_R_0";
    const cldnn::primitive_id crop_id_R_1 = layerName + "_crop_R_1";
    const cldnn::primitive_id crop_id_R_2 = layerName + "_crop_R_2";
    const cldnn::primitive_id crop_id_R_3 = layerName + "_crop_R_3";
    const cldnn::primitive_id concat_id_R = layerName + "_concatR";
    const cldnn::primitive_id permute_R = layerName + "_permuteR";
    const cldnn::primitive_id permute_R2 = layerName + "_permuteR2";
    const cldnn::primitive_id crop_id_B_0 = layerName + "_crop_B_0";
    const cldnn::primitive_id crop_id_B_1 = layerName + "_crop_B_1";
    const cldnn::primitive_id crop_id_B_2 = layerName + "_crop_B_2";
    const cldnn::primitive_id crop_id_B_3 = layerName + "_crop_B_3";
    const cldnn::primitive_id concat_id_B = layerName + "_concatB";
    const cldnn::primitive_id permute_B = layerName + "_permuteB";
    //p.add_primitive(*op, cldnn::permute(permute_id_1, inputs[0], {1, 0, 2, 3}));
    //p.add_primitive(*op, cldnn::permute(permute_id_2, inputs[1], {3, 1, 0, 2}));
    p.add_primitive(*op, cldnn::permute(permute_id_3, inputs[2], {3, 1, 0, 2}));
    const unsigned long int gateNum = 4;
    int hiddenSize = static_cast<int>(op->get_input_shape(4)[1]/gateNum);
    //W
    auto WShape = op->get_input_shape(4);
    cldnn::layout WLayout = cldnn::layout(
                cldnn::element_type_to_data_type(mutable_precision_firstsecond),
                cldnn::format::bfzyx,
                tensor_from_dims({WShape[0], WShape[1], 1, WShape[3], WShape[2]}));
    auto cropSize = cldnn::tensor{1, 1, 1, static_cast<int>(WShape[2]), hiddenSize};
    p.add_primitive(*op, cldnn::reorder(permute_unsqueeze_1, inputs[4], WLayout));
    p.add_primitive(*op, cldnn::permute(permute_W, permute_unsqueeze_1, {0, 4, 2, 3, 1})); //0 1 4 3 2 ->
    p.add_primitive(*op, cldnn::crop(crop_id_W_0, permute_W, cropSize, cldnn::tensor{0, 0, 0, 0, 0}));
    p.add_primitive(*op, cldnn::crop(crop_id_W_1, permute_W, cropSize, cldnn::tensor{0, 0, 0, 0, hiddenSize}));
    p.add_primitive(*op, cldnn::crop(crop_id_W_2, permute_W, cropSize, cldnn::tensor{0, 0, 0, 0, 2*hiddenSize}));
    p.add_primitive(*op, cldnn::crop(crop_id_W_3, permute_W, cropSize, cldnn::tensor{0, 0, 0, 0, 3*hiddenSize}));
    p.add_primitive(*op, cldnn::concatenation(concat_id_W, {crop_id_W_1, crop_id_W_0, crop_id_W_2, crop_id_W_3}, 4));
    p.add_primitive(*op, cldnn::permute(permute_W2, concat_id_W, {0, 1, 3, 4, 2}));
    //R
    auto RShape = op->get_input_shape(5);
    cldnn::layout RLayout = cldnn::layout(
                cldnn::element_type_to_data_type(mutable_precision_firstsecond),
                cldnn::format::bfzyx,
                tensor_from_dims({RShape[0], RShape[1], RShape[2], RShape[3], 1}));
    auto cropSizeR = cldnn::tensor{1, 1, 1, static_cast<int>(RShape[2]), hiddenSize};
    p.add_primitive(*op, cldnn::reorder(permute_unsqueeze_R, inputs[5], RLayout));
    p.add_primitive(*op, cldnn::permute(permute_R, permute_unsqueeze_R, {0, 4, 2, 3, 1}));
    p.add_primitive(*op, cldnn::crop(crop_id_R_0, permute_R, cropSizeR, cldnn::tensor{0, 0, 0, 0, 0}));
    p.add_primitive(*op, cldnn::crop(crop_id_R_1, permute_R, cropSizeR, cldnn::tensor{0, 0, 0, 0, hiddenSize}));
    p.add_primitive(*op, cldnn::crop(crop_id_R_2, permute_R, cropSizeR, cldnn::tensor{0, 0, 0, 0, 2*hiddenSize}));
    p.add_primitive(*op, cldnn::crop(crop_id_R_3, permute_R, cropSizeR, cldnn::tensor{0, 0, 0, 0, 3*hiddenSize}));
    p.add_primitive(*op, cldnn::concatenation(concat_id_R, {crop_id_R_1, crop_id_R_0, crop_id_R_2, crop_id_R_3}, 4));
    p.add_primitive(*op, cldnn::permute(permute_R2, concat_id_R, {0, 1, 3, 4, 2}));
    //B
    auto BShape = op->get_input_shape(6);
    auto cropSizeB = cldnn::tensor{1, hiddenSize, 1, 1};
    p.add_primitive(*op, cldnn::crop(crop_id_B_0, inputs[6], cropSizeB, cldnn::tensor{0, 0, 0, 0}));
    p.add_primitive(*op, cldnn::crop(crop_id_B_1, inputs[6], cropSizeB, cldnn::tensor{0, hiddenSize, 0, 0}));
    p.add_primitive(*op, cldnn::crop(crop_id_B_2, inputs[6], cropSizeB, cldnn::tensor{0, 2*hiddenSize, 0, 0}));
    p.add_primitive(*op, cldnn::crop(crop_id_B_3, inputs[6], cropSizeB, cldnn::tensor{0, 3*hiddenSize, 0, 0}));
    p.add_primitive(*op, cldnn::concatenation(concat_id_B, {crop_id_B_1, crop_id_B_0, crop_id_B_2, crop_id_B_3}, 0));
    p.add_primitive(*op, cldnn::permute(permute_B, concat_id_B, {2, 3, 0, 1}));
    cldnn::lstm_seq prim({lstm_seq_id + ".out_pre_perm", inputs[0], inputs[1], \
        permute_id_3, permute_W2, permute_R2, permute_B, inputs[3], mutable_id_1, mutable_id_2, \
        clip, activations, activation_params, cldnn::lstm_weights_order::fizo, direction});
    p.add_primitive(*op, prim);
    p.add_primitive(*op, cldnn::permute(lstm_seq_id + ".out0", lstm_seq_id + ".out_pre_perm", {1, 3, 0, 2}));
    p.add_primitive(*op, cldnn::mutable_data(lstm_seq_id + ".out1_pre_perm", {cldnn::input_info(lstm_seq_id + ".out_pre_perm")}, shared_memories.front()));
    p.add_primitive(*op, cldnn::mutable_data(lstm_seq_id + ".out2_pre_perm", {cldnn::input_info(lstm_seq_id + ".out_pre_perm")}, shared_memories.back()));
    p.add_primitive(*op, cldnn::permute(lstm_seq_id + ".out1", lstm_seq_id + ".out1_pre_perm", {2, 1, 3, 0}));
    p.add_primitive(*op, cldnn::permute(lstm_seq_id + ".out2", lstm_seq_id + ".out2_pre_perm", {2, 1, 3, 0}));
}

REGISTER_FACTORY_IMPL(v4, LSTMCell);
REGISTER_FACTORY_IMPL(v5, LSTMSequence);

}  // namespace intel_gpu
}  // namespace ov
