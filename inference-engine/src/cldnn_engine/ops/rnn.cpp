// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/lstm_cell.hpp"
#include "ngraph/op/lstm_sequence.hpp"

#include "api/reshape.hpp"
#include "api/reorder.hpp"
#include "api/fully_connected.hpp"
#include "api/lstm.hpp"
#include "api/crop.hpp"
#include "api/concatenation.hpp"

namespace CLDNNPlugin {
cldnn::activation_func GetActivationFunc(std::string name) {
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

void CreateLSTMCellOp(Program& p, const std::shared_ptr<ngraph::op::v4::LSTMCell>& op) {
    p.ValidateInputs(op, {6});
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
    auto lstm_dtype = DataTypeFromPrecision(op->get_output_element_type(0));

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
    p.AddPrimitive(cldnn::reshape(inReshapeID, inputPrimitives[0], inputShape));
    p.AddPrimitive(cldnn::reorder(permuteID, inReshapeID, inputLayout));

    p.AddInnerPrimitiveToProfiler(inReshapeID, op->get_friendly_name(), op);
    p.AddInnerPrimitiveToProfiler(permuteID, op->get_friendly_name(), op);

    std::string hiddenInResh = inHiddenReshapeID + "_1";
    std::string hiddenInStr = inHiddenReorderID + "_1";
    std::string cellInResh = inHiddenReshapeID + "_2";
    std::string cellInStr = inHiddenReorderID + "_2";
    p.AddPrimitive(cldnn::reshape(hiddenInResh, inputPrimitives[1], inStateShape));
    p.AddPrimitive(cldnn::reorder(hiddenInStr, hiddenInResh, hiddenLayout));
    p.AddPrimitive(cldnn::reshape(cellInResh, inputPrimitives[2], inStateShape));
    p.AddPrimitive(cldnn::reorder(cellInStr, cellInResh, hiddenLayout));
    p.AddPrimitive(cldnn::concatenation(input_concatID, { permuteID, hiddenInStr }, cldnn::concatenation::concatenation_axis::along_x));

    p.AddInnerPrimitiveToProfiler(hiddenInResh, op->get_friendly_name(), op);
    p.AddInnerPrimitiveToProfiler(hiddenInStr, op->get_friendly_name(), op);
    p.AddInnerPrimitiveToProfiler(cellInResh, op->get_friendly_name(), op);
    p.AddInnerPrimitiveToProfiler(cellInStr, op->get_friendly_name(), op);
    p.AddInnerPrimitiveToProfiler(input_concatID, op->get_friendly_name(), op);

    cldnn::tensor gemmSz = cldnn::tensor{ lstm_batch_size, 1, 4 * lstm_hidden_size, 1 };
    cldnn::layout gemmLayout = cldnn::layout(lstm_dtype, cldnn::format::bfyx, gemmSz);
    cldnn::tensor hiddenSz = cldnn::tensor{ lstm_batch_size, 1, lstm_hidden_size, 1 };
    cldnn::tensor cellCropSz = cldnn::tensor{0, 1, 0, 0};

    std::string lstm_fc_id = layerName + "_fully_connected";
    std::string lstm_elt_id = layerName + "_lstm_elt";
    std::string crop_id = layerName + "_crop";

    cldnn::primitive_id WRconcatID = layerName + "_WRconcat";
    p.AddPrimitive(cldnn::concatenation(WRconcatID, { weightID, recurrentID }, cldnn::concatenation::concatenation_axis::along_f));
    p.AddInnerPrimitiveToProfiler(WRconcatID, op->get_friendly_name(), op);

    p.AddPrimitive(cldnn::fully_connected(lstm_fc_id, input_concatID, WRconcatID, hasBias ? biasID : ""));
    p.AddPrimitive(cldnn::reshape(gemmReshapeID, lstm_fc_id, gemmSz));
    p.AddPrimitive(cldnn::reorder(gemmReorderID, gemmReshapeID, gemmLayout));
    p.AddPrimitive(cldnn::lstm_elt(lstm_elt_id, gemmReorderID, cellInStr,
                                 clip, 0, activations, activation_params, cldnn::lstm_weights_order::fizo));

    p.AddInnerPrimitiveToProfiler(lstm_fc_id, op->get_friendly_name(), op);
    p.AddInnerPrimitiveToProfiler(gemmReshapeID, op->get_friendly_name(), op);
    p.AddInnerPrimitiveToProfiler(gemmReorderID, op->get_friendly_name(), op);
    p.AddInnerPrimitiveToProfiler(lstm_elt_id, op->get_friendly_name(), op);

    cldnn::tensor outSz = cldnn::tensor{ lstm_batch_size, lstm_hidden_size, 1, 1 };
    cldnn::primitive_id outputHiddenCropID = layerName + "_hc";
    cldnn::primitive_id outputHiddenID = layerName + ".0";
    p.AddPrimitive(cldnn::crop(outputHiddenCropID, lstm_elt_id, hiddenSz, cldnn::tensor{0, 0, 0, 0}));
    p.AddInnerPrimitiveToProfiler(outputHiddenCropID, op->get_friendly_name(), op);
    p.AddPrimitive(cldnn::reshape(outputHiddenID, outputHiddenCropID, outSz));
    p.AddInnerPrimitiveToProfiler(outputHiddenID, op->get_friendly_name(), op);

    cldnn::primitive_id outputCellCropID = layerName + "_cc";
    cldnn::primitive_id outputCellID = layerName + ".1";
    p.AddPrimitive(cldnn::crop(outputCellCropID, lstm_elt_id, hiddenSz, cellCropSz));
    p.AddInnerPrimitiveToProfiler(outputCellCropID, op->get_friendly_name(), op);
    p.AddPrimitive(cldnn::reshape(outputCellID, outputCellCropID, outSz));
    p.AddInnerPrimitiveToProfiler(outputCellID, op->get_friendly_name(), op);

    // output primitive IDs
    p.primitiveIDs[outputHiddenID] = outputHiddenID;     // LSTMCell:LSTMCell - "concat hidden"
    p.primitiveIDs[layerName] = outputHiddenID;          // LSTMCell:LSTMCell:0 - hidden state
    p.primitiveIDs[outputCellID] = outputCellID;         // LSTMCell:LSTMCell:1 - cell state

    p.AddPrimitiveToProfiler(layerName, op, outputHiddenID);
}

void CreateLSTMSequenceOp(Program& p, const std::shared_ptr<ngraph::op::v5::LSTMSequence>& op) {
    p.ValidateInputs(op, {7});

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
    auto lstm_dtype = DataTypeFromPrecision(op->get_output_element_type(0));

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
    p.AddPrimitive(cldnn::reshape(inReshapeID, inputPrimitives[0], inputShape));
    p.AddPrimitive(cldnn::reorder(permuteID, inReshapeID, inputLayout));

    p.AddPrimitive(cldnn::reshape(inHiddenStateID, inputPrimitives[1], inStateShape));
    p.AddPrimitive(cldnn::reshape(inCellStateID, inputPrimitives[2], inStateShape));

    p.AddInnerPrimitiveToProfiler(inReshapeID, op->get_friendly_name(), op);
    p.AddInnerPrimitiveToProfiler(permuteID, op->get_friendly_name(), op);
    p.AddInnerPrimitiveToProfiler(inHiddenStateID, op->get_friendly_name(), op);
    p.AddInnerPrimitiveToProfiler(inCellStateID, op->get_friendly_name(), op);

    cldnn::tensor gemmSz = cldnn::tensor{ lstm_batch_size, 1, 4 * lstm_hidden_size, 1 };
    cldnn::layout gemmLayout = cldnn::layout(lstm_dtype, cldnn::format::bfyx, gemmSz);
    cldnn::tensor hiddenSz = cldnn::tensor{ lstm_batch_size, 1, lstm_hidden_size, 1 };
    cldnn::tensor cellCropSz = cldnn::tensor{0, 1, 0, 0};
    cldnn::primitive_id hiddenStr = inHiddenReshapeID + "_1";
    cldnn::primitive_id cellStr = inHiddenReshapeID + "_2";
    cldnn::primitive_id inputCropID = layerName + "_inputCrop";

    cldnn::primitive_id WRconcatID = layerName + "_WRconcat";
    p.AddPrimitive(cldnn::concatenation(WRconcatID, { weightID, recurrentID }, cldnn::concatenation::concatenation_axis::along_y));
    p.AddInnerPrimitiveToProfiler(WRconcatID, op->get_friendly_name(), op);

    std::vector<size_t> WRreshapeSize = { 4 * size_t(lstm_hidden_size), size_t(lstm_input_size + lstm_hidden_size) };
    cldnn::primitive_id WRreshapeID = WRconcatID + "_reshape";
    auto reshapeInPrim = cldnn::reshape(WRreshapeID, WRconcatID, CldnnTensorFromIEDims(WRreshapeSize));
    p.AddPrimitive(reshapeInPrim);
    p.AddInnerPrimitiveToProfiler(WRreshapeID, op->get_friendly_name(), op);

    for (int i = 0; i < lstm_sequence_len; ++i) {
        const std::string id_str = std::to_string(i);
        cldnn::primitive_id concatID = layerName + "_inputConcat" + id_str;
        cldnn::primitive_id lstm_fc_id = layerName + "_fully_connected" + id_str;
        cldnn::primitive_id lstm_fc_resh_id = layerName + "_gemmReshape" + id_str;
        cldnn::primitive_id lstm_fc_reor_id = layerName + "_gemmReorder" + id_str;
        cldnn::primitive_id lstm_elt_id = layerName + "_lstm_elt" + id_str;
        cldnn::primitive_id crop_id = layerName + "_crop" + id_str;

        int seqIdx = isForward ? i : lstm_sequence_len - 1 - i;
        const std::string seqIdx_str = std::to_string(seqIdx);

        cldnn::tensor crop_tensor{ inputShape.batch[0], 1, inputShape.spatial[0], inputShape.spatial[1] };
        cldnn::tensor offset_tensor{ 0, static_cast<cldnn::tensor::value_type>(seqIdx), 0, 0 };
        cldnn::primitive_id inputCrop_id = inputCropID + ":" + seqIdx_str;
        p.AddPrimitive(cldnn::crop(inputCrop_id, permuteID, crop_tensor, offset_tensor));
        p.AddInnerPrimitiveToProfiler(inputCrop_id, op->get_friendly_name(), op);

        p.AddPrimitive(cldnn::concatenation(concatID, { inputCrop_id, hiddenStr }, cldnn::concatenation::concatenation_axis::along_x));
        p.AddInnerPrimitiveToProfiler(concatID, op->get_friendly_name(), op);
        p.AddPrimitive(cldnn::fully_connected(lstm_fc_id, concatID, WRreshapeID, biasID));
        p.AddInnerPrimitiveToProfiler(lstm_fc_id, op->get_friendly_name(), op);

        p.AddPrimitive(cldnn::reshape(lstm_fc_resh_id, lstm_fc_id, gemmSz));
        p.AddPrimitive(cldnn::reorder(lstm_fc_reor_id, lstm_fc_resh_id, gemmLayout));
        p.AddPrimitive(cldnn::lstm_elt(lstm_elt_id, lstm_fc_reor_id, cellStr,
                                     clip, 0, activations, activation_params, cldnn::lstm_weights_order::fizo));
        p.AddInnerPrimitiveToProfiler(lstm_fc_resh_id, op->get_friendly_name(), op);
        p.AddInnerPrimitiveToProfiler(lstm_fc_reor_id, op->get_friendly_name(), op);
        p.AddInnerPrimitiveToProfiler(lstm_elt_id, op->get_friendly_name(), op);

        hiddenStr = crop_id + ":hidden";
        cellStr = crop_id + ":cell";
        p.AddPrimitive(cldnn::crop(hiddenStr, lstm_elt_id, hiddenSz, cldnn::tensor{ 0, 0, 0, 0 }));
        p.AddInnerPrimitiveToProfiler(hiddenStr, op->get_friendly_name(), op);
        output_ids_offsets.push_back(hiddenStr);

        if (i < lstm_sequence_len - 1) {
            p.AddPrimitive(cldnn::crop(cellStr, lstm_elt_id, hiddenSz, cellCropSz));
            p.AddInnerPrimitiveToProfiler(cellStr, op->get_friendly_name(), op);
        } else {
            // last hidden state crop (output 2)
            cldnn::primitive_id outputHiddenID = layerName + ".1";
            p.primitiveIDs[hiddenStr] = hiddenStr;
            p.primitiveIDs[outputHiddenID] = hiddenStr;

            // last cell state crop (output 3)
            p.AddPrimitive(cldnn::crop(cellStr, lstm_elt_id, hiddenSz, cellCropSz));
            cldnn::primitive_id outputCellID = layerName + ".2";
            p.AddInnerPrimitiveToProfiler(cellStr, op->get_friendly_name(), op);
            p.primitiveIDs[outputCellID] = cellStr;
        }
    }

    if (!isForward) std::reverse(output_ids_offsets.begin(), output_ids_offsets.end());
    // concatenated hidden state (output 1)
    cldnn::primitive_id outputConcatID = layerName + ".0";
    cldnn::primitive_id concatStr = layerName + ":hiddenConcat";
    p.AddPrimitive(cldnn::concatenation(concatStr, output_ids_offsets, cldnn::concatenation::along_f));

    p.primitiveIDs[outputConcatID] = concatStr;
    p.primitiveIDs[layerName] = concatStr;
    p.AddPrimitiveToProfiler(layerName, op);
}

REGISTER_FACTORY_IMPL(v4, LSTMCell);
REGISTER_FACTORY_IMPL(v5, LSTMSequence);

}  // namespace CLDNNPlugin
