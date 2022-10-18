// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/lstm_cell.hpp"
#include "ngraph/op/lstm_sequence.hpp"

#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/lstm.hpp"
#include "intel_gpu/primitives/crop.hpp"
#include "intel_gpu/primitives/concatenation.hpp"

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
            IE_THROW() << "Wrong number of activations for LSTMCell op " << op->get_friendly_name();
        for (int i = 0; i < 3; i++) {
            auto af = GetActivationFunc(op_activations[i]);
            if (af == cldnn::activation_func::none)
                IE_THROW() << "Wrong or unsupported activation type " << op_activations[i]
                << " for LSTMCell op " << op->get_friendly_name();
            activations[i] = af;
        }
    }
    auto op_a = op->get_activations_alpha();
    auto op_b = op->get_activations_beta();
    if (!op_a.empty()) {
        if (op_a.size() != 3 || op_b.size() != 3)
            IE_THROW() << "Wrong number of activation parameters for LSTMCell op " << op->get_friendly_name();
        for (int i = 0; i < 3; i++) {
            cldnn::activation_additional_params params = { op_a[i], op_b[i] };
            activation_params.push_back(cldnn::activation_additional_params(params));
        }
    }
}

static void CreateLSTMCellOp(Program& p, const std::shared_ptr<ngraph::op::v4::LSTMCell>& op) {
    validate_inputs_count(op, {6});
    int lstm_batch_size, lstm_input_size, lstm_hidden_size;
    bool hasBias = true;
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);

    std::string layerName = layer_type_name_ID(op);
    cldnn::primitive_id weightID = inputPrimitives[3];
    cldnn::primitive_id recurrentID = inputPrimitives[4];
    cldnn::primitive_id biasID = inputPrimitives[5];

    /* check incoming CNN layer and setup required variables */
    {
        const auto in_dims0 = op->get_input_shape(0);
        const auto out_dims0 = op->get_output_shape(0);

        if (in_dims0.size() != 2 ||
            op->get_input_shape(1).size() != 2 ||
            op->get_input_shape(2).size() != 2)
            IE_THROW() << "Wrong input shapes for LSTMCell op " << op->get_friendly_name();

        lstm_input_size = in_dims0.back();
        lstm_batch_size = in_dims0.at(in_dims0.size()-2);
        lstm_hidden_size = out_dims0.back();
    }

    std::vector<cldnn::activation_func> activations;
    std::vector<cldnn::activation_additional_params> activation_params;
    GetLSTMActivationParams(op, activations, activation_params);
    float clip = op->get_clip();

    //  LSTM primitive works with single precision for all in/out/weights tensors
    auto lstm_dtype = cldnn::element_type_to_data_type(op->get_output_element_type(0));

    cldnn::primitive_id inReshapeID = layerName + "_inReshape";
    cldnn::primitive_id permuteID = layerName + "_inputReorder";
    cldnn::primitive_id inHiddenReshapeID = layerName + "_inHiddenReshape";
    cldnn::primitive_id inHiddenReorderID = layerName + "_inHiddenReorder";
    cldnn::primitive_id gemmReshapeID = layerName + "_gemmReshape";
    cldnn::primitive_id gemmReorderID = layerName + "_gemmReorder";
    cldnn::primitive_id input_concatID = layerName + "_inputConcat";

    cldnn::tensor inputShape = { lstm_batch_size, 1, lstm_input_size, 1 };
    cldnn::tensor inStateShape = { lstm_batch_size, 1, lstm_hidden_size, 1 };
    cldnn::layout inputLayout = cldnn::layout(lstm_dtype, cldnn::format::bfyx, inputShape);
    cldnn::layout hiddenLayout = cldnn::layout(lstm_dtype, cldnn::format::bfyx, inStateShape);
    p.add_primitive(*op, cldnn::reshape(inReshapeID, inputPrimitives[0], inputShape));
    p.add_primitive(*op, cldnn::reorder(permuteID, inReshapeID, inputLayout));


    std::string hiddenInResh = inHiddenReshapeID + "_1";
    std::string hiddenInStr = inHiddenReorderID + "_1";
    std::string cellInResh = inHiddenReshapeID + "_2";
    std::string cellInStr = inHiddenReorderID + "_2";
    p.add_primitive(*op, cldnn::reshape(hiddenInResh, inputPrimitives[1], inStateShape));
    p.add_primitive(*op, cldnn::reorder(hiddenInStr, hiddenInResh, hiddenLayout));
    p.add_primitive(*op, cldnn::reshape(cellInResh, inputPrimitives[2], inStateShape));
    p.add_primitive(*op, cldnn::reorder(cellInStr, cellInResh, hiddenLayout));
    p.add_primitive(*op, cldnn::concatenation(input_concatID,
                                              { permuteID, hiddenInStr },
                                              3));

    cldnn::tensor gemmSz = cldnn::tensor{ lstm_batch_size, 1, 4 * lstm_hidden_size, 1 };
    cldnn::layout gemmLayout = cldnn::layout(lstm_dtype, cldnn::format::bfyx, gemmSz);
    cldnn::tensor hiddenSz = cldnn::tensor{ lstm_batch_size, 1, lstm_hidden_size, 1 };
    cldnn::tensor cellCropSz = cldnn::tensor{0, 1, 0, 0};

    std::string lstm_fc_id = layerName + "_fully_connected";
    std::string lstm_elt_id = layerName + "_lstm_elt";
    std::string crop_id = layerName + "_crop";

    cldnn::primitive_id WRconcatID = layerName + "_WRconcat";
    p.add_primitive(*op, cldnn::concatenation(WRconcatID, { weightID, recurrentID }, 1));

    cldnn::primitive_id FCInputReshapeID = "Reshape_bf_" + lstm_fc_id + "_for_input";
    cldnn::tensor FCInputReshapeSz = { lstm_batch_size, inputShape.spatial[0] + inStateShape.spatial[0], 1, 1 };
    p.add_primitive(*op, cldnn::reshape(FCInputReshapeID, input_concatID, FCInputReshapeSz));

    p.add_primitive(*op, cldnn::fully_connected(lstm_fc_id, FCInputReshapeID, WRconcatID, hasBias ? biasID : ""));
    p.add_primitive(*op, cldnn::reshape(gemmReshapeID, lstm_fc_id, gemmSz));
    p.add_primitive(*op, cldnn::reorder(gemmReorderID, gemmReshapeID, gemmLayout));
    p.add_primitive(*op, cldnn::lstm_elt(lstm_elt_id, gemmReorderID, cellInStr, clip, 0, activations,
                                         activation_params, cldnn::lstm_weights_order::fizo, 0));


    cldnn::tensor outSz = cldnn::tensor{ lstm_batch_size, lstm_hidden_size, 1, 1 };
    cldnn::primitive_id outputHiddenCropID = layerName + "_hc";
    cldnn::primitive_id outputHiddenID = layerName + ".out0";
    p.add_primitive(*op, cldnn::crop(outputHiddenCropID, lstm_elt_id, hiddenSz, cldnn::tensor{0, 0, 0, 0}));
    p.add_primitive(*op, cldnn::reshape(outputHiddenID, outputHiddenCropID, outSz), {layerName});

    cldnn::primitive_id outputCellCropID = layerName + "_cc";
    cldnn::primitive_id outputCellID = layerName + ".out1";
    p.add_primitive(*op, cldnn::crop(outputCellCropID, lstm_elt_id, hiddenSz, cellCropSz));
    p.add_primitive(*op, cldnn::reshape(outputCellID, outputCellCropID, outSz));
}

static void CreateLSTMSequenceOp(Program& p, const std::shared_ptr<ngraph::op::v5::LSTMSequence>& op) {
    validate_inputs_count(op, {7});

    std::string layerName = layer_type_name_ID(op);
    int lstm_batch_size, lstm_input_size, lstm_hidden_size, lstm_sequence_len;

    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    cldnn::primitive_id weightID = inputPrimitives[4];
    cldnn::primitive_id recurrentID = inputPrimitives[5];
    cldnn::primitive_id biasID = inputPrimitives[6];

    {
        const auto in_dims0 = op->get_input_shape(0);
        const auto out_dims0 = op->get_output_shape(0);

        if (in_dims0.size() != 3 ||
            op->get_input_shape(1).size() != 3 ||
            op->get_input_shape(2).size() != 3)
            IE_THROW() << "Wrong input shapes for LSTMSequence op " << op->get_friendly_name();

        lstm_input_size = in_dims0.back();
        lstm_sequence_len = in_dims0.at(in_dims0.size() - 2);
        lstm_batch_size = in_dims0.at(in_dims0.size() - 3);
        lstm_hidden_size = out_dims0.back();
    }

    std::vector<cldnn::activation_func> activations;
    std::vector<cldnn::activation_additional_params> activation_params;
    GetLSTMActivationParams(op, activations, activation_params);
    float clip = op->get_clip();
    bool isForward = op->get_direction() == ngraph::op::RecurrentSequenceDirection::FORWARD;

    //  LSTM primitive works with single precision for all in/out/weights tensors
    auto lstm_dtype = cldnn::element_type_to_data_type(op->get_output_element_type(0));

    cldnn::primitive_id inReshapeID = layerName + "_inReshape";
    cldnn::primitive_id permuteID = layerName + "_inputReorder";
    cldnn::primitive_id inHiddenReshapeID = layerName + "_inHiddenReshape";
    cldnn::primitive_id inHiddenReorderID = layerName + "_inHiddenReorder";
    cldnn::primitive_id inHiddenStateID = inHiddenReshapeID + "_1";
    cldnn::primitive_id inCellStateID = inHiddenReshapeID + "_2";

    std::vector<cldnn::primitive_id> output_ids_offsets;

    cldnn::tensor inputShape = { lstm_batch_size, lstm_sequence_len, lstm_input_size, 1 };
    cldnn::tensor inStateShape = { lstm_batch_size, 1, lstm_hidden_size, 1 };
    cldnn::layout inputLayout = cldnn::layout(lstm_dtype, cldnn::format::bfyx, inputShape);
    p.add_primitive(*op, cldnn::reshape(inReshapeID, inputPrimitives[0], inputShape));
    p.add_primitive(*op, cldnn::reorder(permuteID, inReshapeID, inputLayout));

    p.add_primitive(*op, cldnn::reshape(inHiddenStateID, inputPrimitives[1], inStateShape));
    p.add_primitive(*op, cldnn::reshape(inCellStateID, inputPrimitives[2], inStateShape));

    cldnn::tensor gemmSz = cldnn::tensor{ lstm_batch_size, 1, 4 * lstm_hidden_size, 1 };
    cldnn::layout gemmLayout = cldnn::layout(lstm_dtype, cldnn::format::bfyx, gemmSz);
    cldnn::tensor hiddenSz = cldnn::tensor{ lstm_batch_size, 1, lstm_hidden_size, 1 };
    cldnn::tensor cellCropSz = cldnn::tensor{0, 1, 0, 0};
    cldnn::primitive_id hiddenStr = inHiddenReshapeID + "_1";
    cldnn::primitive_id cellStr = inHiddenReshapeID + "_2";
    cldnn::primitive_id inputCropID = layerName + "_inputCrop";

    cldnn::primitive_id WRconcatID = layerName + "_WRconcat";
    p.add_primitive(*op, cldnn::concatenation(WRconcatID, { weightID, recurrentID }, 2));

    std::vector<size_t> WRreshapeSize = { 4 * size_t(lstm_hidden_size), size_t(lstm_input_size + lstm_hidden_size) };
    cldnn::primitive_id WRreshapeID = WRconcatID + "_reshape";
    auto reshapeInPrim = cldnn::reshape(WRreshapeID, WRconcatID, tensor_from_dims(WRreshapeSize));
    p.add_primitive(*op, reshapeInPrim);

    for (int i = 0; i < lstm_sequence_len; ++i) {
        const std::string id_str = std::to_string(i);
        cldnn::primitive_id concatID = layerName + "_inputConcat" + id_str;
        cldnn::primitive_id lstm_fc_id = layerName + "_fully_connected" + id_str;
        cldnn::primitive_id fc_input_resh_id = "Reshape_bf_" + lstm_fc_id + "_for_input" + id_str;
        cldnn::primitive_id lstm_fc_resh_id = layerName + "_gemmReshape" + id_str;
        cldnn::primitive_id lstm_fc_reor_id = layerName + "_gemmReorder" + id_str;
        cldnn::primitive_id lstm_elt_id = layerName + "_lstm_elt" + id_str;
        cldnn::primitive_id crop_id = layerName + "_crop" + id_str;

        int seqIdx = isForward ? i : lstm_sequence_len - 1 - i;
        const std::string seqIdx_str = std::to_string(seqIdx);

        cldnn::tensor crop_tensor{ inputShape.batch[0], 1, inputShape.spatial[0], inputShape.spatial[1] };
        cldnn::tensor offset_tensor{ 0, static_cast<cldnn::tensor::value_type>(seqIdx), 0, 0 };
        cldnn::primitive_id inputCrop_id = inputCropID + ":" + seqIdx_str;
        p.add_primitive(*op, cldnn::crop(inputCrop_id, permuteID, crop_tensor, offset_tensor));

        p.add_primitive(*op, cldnn::concatenation(concatID, { inputCrop_id, hiddenStr }, 3));

        cldnn::tensor fc_input_resh_tensor = { crop_tensor.batch[0], crop_tensor.spatial[0] + inStateShape.spatial[0],
                                               crop_tensor.feature[0], crop_tensor.spatial[1]};
        p.add_primitive(*op, cldnn::reshape(fc_input_resh_id, concatID, fc_input_resh_tensor));

        p.add_primitive(*op, cldnn::fully_connected(lstm_fc_id, fc_input_resh_id, WRreshapeID, biasID));

        p.add_primitive(*op, cldnn::reshape(lstm_fc_resh_id, lstm_fc_id, gemmSz));
        p.add_primitive(*op, cldnn::reorder(lstm_fc_reor_id, lstm_fc_resh_id, gemmLayout));
        p.add_primitive(*op, cldnn::lstm_elt(lstm_elt_id, lstm_fc_reor_id, cellStr, clip, 0, activations,
                                             activation_params, cldnn::lstm_weights_order::fizo, 0));

        hiddenStr = crop_id + ":hidden";
        cellStr = crop_id + ":cell";
        cldnn::primitive_id outputHiddenID = layerName + ".out1";
        p.add_primitive(*op, cldnn::crop(hiddenStr, lstm_elt_id, hiddenSz, cldnn::tensor{ 0, 0, 0, 0 }), {outputHiddenID});
        output_ids_offsets.push_back(hiddenStr);

        if (i < lstm_sequence_len - 1) {
            p.add_primitive(*op, cldnn::crop(cellStr, lstm_elt_id, hiddenSz, cellCropSz));
        } else {
            // last hidden state crop (output 2)

            // last cell state crop (output 3)
            cldnn::primitive_id outputCellID = layerName + ".out2";
            p.add_primitive(*op, cldnn::crop(cellStr, lstm_elt_id, hiddenSz, cellCropSz), {outputCellID});
        }
    }

    if (!isForward) std::reverse(output_ids_offsets.begin(), output_ids_offsets.end());
    // concatenated hidden state (output 1)
    cldnn::primitive_id outputConcatID = layerName + ".out0";
    cldnn::primitive_id concatStr = layerName + ":hiddenConcat";
    p.add_primitive(*op, cldnn::concatenation(concatStr, output_ids_offsets, 1), {outputConcatID, layerName});
}

REGISTER_FACTORY_IMPL(v4, LSTMCell);
REGISTER_FACTORY_IMPL(v5, LSTMSequence);

}  // namespace intel_gpu
}  // namespace ov
