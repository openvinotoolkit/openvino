// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <sstream>
#include <utility>
#include <api/cldnn.hpp>
#include <api/data.hpp>
#include <api/mutable_data.hpp>
#include <api/reorder.hpp>
#include <api/fully_connected.hpp>
#include <api/concatenation.hpp>
#include <api/reshape.hpp>
#include <api/permute.hpp>
#include <api/split.hpp>
#include <api/crop.hpp>
#include <api/reverse_sequence.hpp>
#include <api/lstm.hpp>
#include <api/lstm_dynamic.hpp>
#include "cldnn_common_utils.h"
#include "cldnn_program.h"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace CLDNNPlugin {

std::string get_string_id(size_t i) {
    std::stringstream ss;
    ss << std::setw(5) << std::setfill('0') << i;
    return ss.str();
}

void Program::CreateLSTMCellPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer) {
    int lstm_batch_size, lstm_input_size, lstm_hidden_size;
    bool hasBias = false;
    auto inputPrimitives = GetPrevLayersPrimitives(layer);

    std::string layerName = layer_type_name_ID(layer);
    cldnn::primitive_id weightID = layerName + m_weightsTag;
    cldnn::primitive_id biasID = layerName + m_biasesTag;

    /* check incoming CNN layer and setup required variables */
    {
        auto in_data0 = layer->insData[0].lock();
        if (!in_data0)
            THROW_IE_EXCEPTION << "Missing first input for LSTMCell layer " << layer->name;

        const auto in_dims0 = in_data0->getTensorDesc().getDims();
        const auto out_dims0 = layer->outData[0]->getTensorDesc().getDims();

        lstm_input_size = in_dims0.back();
        lstm_batch_size = in_dims0.at(in_dims0.size()-2);
        lstm_hidden_size = out_dims0.back();

        auto in_data1 = layer->insData[1].lock();
        if (!in_data1)
            THROW_IE_EXCEPTION << "Missing second input for LSTMCell layer " << layer->name;

        auto in_data2 = layer->insData[2].lock();
        if (!in_data2)
            THROW_IE_EXCEPTION << "Missing third input for LSTMCell layer " << layer->name;

        if (in_dims0.size() != 2 ||
            in_data1->getTensorDesc().getDims().size() != 2 ||
            in_data2->getTensorDesc().getDims().size() != 2)
            THROW_IE_EXCEPTION << "Wrong input shapes for LSTMCell Layer " << layer->name;
    }

    /* Prepare weight/bias memory primitives */
    {
        auto wLayer = as<InferenceEngine::WeightableLayer *>(layer);
        auto pWeightsBlob = wLayer->_weights;
        cldnn::tensor wTensor = cldnn::tensor(cldnn::batch(4 * lstm_hidden_size), cldnn::feature(1), cldnn::spatial(lstm_input_size + lstm_hidden_size, 1));
        cldnn::layout WLayout = cldnn::layout(DataTypeFromPrecision(pWeightsBlob->getTensorDesc().getPrecision()), m_defaultFormat, wTensor);
        weightID = CreatePrimitiveFromBlob(topology, weightID, pWeightsBlob, WLayout);

        /* create bias memory primitive */
        auto pBiasBlob = wLayer->_biases;
        if (pBiasBlob != nullptr) {
            cldnn::tensor bTensor = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(4 * lstm_hidden_size, 1));
            cldnn::layout BLayout = cldnn::layout(DataTypeFromPrecision(pBiasBlob->getTensorDesc().getPrecision()), m_defaultFormat, bTensor);

            biasID = CreatePrimitiveFromBlob(topology, biasID, pBiasBlob, BLayout);
            hasBias = true;
        }
    }

    cldnn::primitive_id inReshapeID = layerName + "_inReshape";
    cldnn::primitive_id permuteID = layerName + "_inputReorder";
    cldnn::primitive_id inHiddenReshapeID = layerName + "_inHiddenReshape";
    cldnn::primitive_id inHiddenReorderID = layerName + "_inHiddenReorder";
    cldnn::primitive_id gemmReshapeID = layerName + "_gemmReshape";
    cldnn::primitive_id gemmReorderID = layerName + "_gemmReorder";
    cldnn::primitive_id concatID = layerName + "_inputConcat";

    //  LSTM primitive works with single precision for all in/out/weights tensors
    auto lstmPrecision = layer->outData[0]->getPrecision();

    cldnn::tensor inputShape = { lstm_batch_size, 1, lstm_input_size, 1 };
    cldnn::tensor hiddenStateShape = { lstm_batch_size, 1, lstm_hidden_size, 1 };
    cldnn::layout inputLayout = cldnn::layout(DataTypeFromPrecision(lstmPrecision), cldnn::format::bfyx, inputShape);
    cldnn::layout hiddenLayout = cldnn::layout(DataTypeFromPrecision(lstmPrecision), cldnn::format::bfyx, hiddenStateShape);
    topology.add(cldnn::reshape(inReshapeID, inputPrimitives[0], inputShape));
    topology.add(cldnn::reorder(permuteID, inReshapeID, inputLayout));

    addInnerPrimitiveToProfiler(inReshapeID, layer->name, layer);
    addInnerPrimitiveToProfiler(permuteID, layer->name, layer);

    std::string hiddenInResh = inHiddenReshapeID + "_1";
    std::string hiddenInStr = inHiddenReorderID + "_1";
    std::string cellInResh = inHiddenReshapeID + "_2";
    std::string cellInStr = inHiddenReorderID + "_2";
    topology.add(cldnn::reshape(hiddenInResh, inputPrimitives[1], hiddenStateShape));
    topology.add(cldnn::reorder(hiddenInStr, hiddenInResh, hiddenLayout));
    topology.add(cldnn::reshape(cellInResh, inputPrimitives[2], hiddenStateShape));
    topology.add(cldnn::reorder(cellInStr, cellInResh, hiddenLayout));
    topology.add(cldnn::concatenation(concatID, { permuteID, hiddenInStr }, cldnn::concatenation::concatenation_axis::along_x));

    addInnerPrimitiveToProfiler(hiddenInResh, layer->name, layer);
    addInnerPrimitiveToProfiler(hiddenInStr, layer->name, layer);
    addInnerPrimitiveToProfiler(cellInResh, layer->name, layer);
    addInnerPrimitiveToProfiler(cellInStr, layer->name, layer);
    addInnerPrimitiveToProfiler(concatID, layer->name, layer);

    cldnn::tensor gemmSz = cldnn::tensor{ lstm_batch_size, 1, 4 * lstm_hidden_size, 1 };
    cldnn::layout gemmLayout = cldnn::layout(DataTypeFromPrecision(lstmPrecision), cldnn::format::bfyx, gemmSz);
    cldnn::tensor hiddenSz = cldnn::tensor{ lstm_batch_size, 1, lstm_hidden_size, 1 };
    cldnn::tensor cellCropSz = cldnn::tensor{0, 1, 0, 0};

    std::string lstm_fc_id = layerName + "_fully_connected";
    std::string lstm_elt_id = layerName + "_lstm_elt";
    std::string crop_id = layerName + "_crop";

    topology.add(cldnn::fully_connected(lstm_fc_id, concatID, weightID, hasBias ? biasID : ""));
    topology.add(cldnn::reshape(gemmReshapeID, lstm_fc_id, gemmSz));
    topology.add(cldnn::reorder(gemmReorderID, gemmReshapeID, gemmLayout));
    topology.add(cldnn::lstm_elt(lstm_elt_id, gemmReorderID, cellInStr,
                                    0, 0, {}, {}, cldnn::lstm_weights_order::fizo));

    addInnerPrimitiveToProfiler(lstm_fc_id, layer->name, layer);
    addInnerPrimitiveToProfiler(gemmReshapeID, layer->name, layer);
    addInnerPrimitiveToProfiler(gemmReorderID, layer->name, layer);
    addInnerPrimitiveToProfiler(lstm_elt_id, layer->name, layer);

    cldnn::primitive_id outputHiddenID = layerName;
    topology.add(cldnn::crop(outputHiddenID, lstm_elt_id, hiddenSz, cldnn::tensor{0, 0, 0, 0}));
    addInnerPrimitiveToProfiler(outputHiddenID, layer->name, layer);
    cldnn::primitive_id outputCellID = layer_type_lower(layer) + ":" + layer->outData[1]->getName();
    topology.add(cldnn::crop(outputCellID, lstm_elt_id, hiddenSz, cellCropSz));
    addInnerPrimitiveToProfiler(outputCellID, layer->name, layer);

    // output primitive IDs
    primitiveIDs[outputHiddenID] = outputHiddenID;                                // LSTMCell:LSTMCell - "concat hidden"
    primitiveIDs[layer_type_lower(layer) + ":" + layer->outData[0]->getName()] = outputHiddenID;   // LSTMCell:LSTMCell:0 - hidden state
    primitiveIDs[outputCellID] = outputCellID;                                    // LSTMCell:LSTMCell:1 - cell state

    addPrimitiveToProfiler(layerName, layer, outputHiddenID);
}

void Program::CreateRegularLSTM(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer) {
    int lstm_batch_size, lstm_sequence_len, lstm_input_size, lstm_hidden_size;
    bool hasInitialHidden = false, hasInitialCell = false, hasBias = false, isForward = true;
    auto inputPrimitives = GetPrevLayersPrimitives(layer);

    std::string layerName = layer_type_name_ID(layer);
    cldnn::primitive_id weightID = layerName + m_weightsTag;
    cldnn::primitive_id biasID = layerName + m_biasesTag;
    auto rnnLayer = as<RNNSequenceLayer*> (layer);
    bool permute_input = (1 != rnnLayer->axis);

    /* check incoming CNN layer and setup required variables */
    {
        if (rnnLayer->cellType != RNNSequenceLayer::LSTM)
         THROW_IE_EXCEPTION << "RNN layer supports only LSTM like cell";

        auto in_data0 = layer->insData[0].lock();
        if (!in_data0)
            THROW_IE_EXCEPTION << "Missing first input for RNN layer " << layer->name;

        const auto in_dims0 = in_data0->getTensorDesc().getDims();
        const auto out_dims0 = layer->outData[0]->getTensorDesc().getDims();

        /* do we have initial hidden and cell?
        if blobs are not null, direct the data from them
        into corresponding LSTM inputs */
        auto in_data1 = layer->insData[1].lock();
        if (in_data1) {
            hasInitialHidden = true;
        }

        auto in_data2 = layer->insData[2].lock();
        if (in_data2) {
            hasInitialCell = true;
        }

        if (in_dims0.size() != 3 ||
            in_data1->getTensorDesc().getDims().size() != 2 ||
            in_data2->getTensorDesc().getDims().size() != 2)
            THROW_IE_EXCEPTION << "Wrong input shapes for RNN Layer " << layer->name;

        if (!permute_input) {
            lstm_batch_size = in_dims0.front();
            lstm_sequence_len = in_dims0[1];
        } else {
            lstm_batch_size = in_dims0[1];
            lstm_sequence_len = in_dims0.front();
        }

        lstm_input_size = in_dims0.back();
        lstm_hidden_size = out_dims0.back();

        if (rnnLayer->direction != RNNSequenceLayer::FWD && rnnLayer->direction != RNNSequenceLayer::BWD)
            THROW_IE_EXCEPTION << "Support only forward and backward direction for RNN Layer " << layer->name;
        isForward = rnnLayer->direction == RNNSequenceLayer::FWD;
    }

    /* Prepare weight/bias memory primitives */
    {
        auto wLayer = as<InferenceEngine::WeightableLayer *>(layer);
        auto pWeightsBlob = wLayer->_weights;
        cldnn::tensor wTensor = cldnn::tensor(cldnn::batch(4 * lstm_hidden_size), cldnn::feature(1), cldnn::spatial(lstm_input_size + lstm_hidden_size, 1));
        cldnn::layout WLayout = cldnn::layout(DataTypeFromPrecision(pWeightsBlob->getTensorDesc().getPrecision()), m_defaultFormat, wTensor);
        weightID = CreatePrimitiveFromBlob(topology, weightID, pWeightsBlob, WLayout);

        /* create bias memory primitive */
        auto pBiasBlob = wLayer->_biases;
        if (pBiasBlob != nullptr) {
            cldnn::tensor bTensor = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(4 * lstm_hidden_size, 1));
            cldnn::layout BLayout = cldnn::layout(DataTypeFromPrecision(pBiasBlob->getTensorDesc().getPrecision()), m_defaultFormat, bTensor);

            biasID = CreatePrimitiveFromBlob(topology, biasID, pBiasBlob, BLayout);
            hasBias = true;
        }
    }

    std::vector<std::pair<cldnn::primitive_id, cldnn::tensor>> input_ids_offsets;
    std::vector<cldnn::primitive_id> output_ids_offsets;

    cldnn::primitive_id inReshapeID = layerName + "_inReshape";
    cldnn::primitive_id permuteID = layerName + "_inputReorder";
    cldnn::primitive_id inHiddenReshapeID = layerName + "_inHiddenReshape";

    //  LSTM primitive works with single precision for all in/out/weights tensors
    auto lstmPrecision = layer->outData[0]->getPrecision();

    cldnn::tensor inputShape;

    if (permute_input) {
        inputShape = { lstm_sequence_len, lstm_batch_size, lstm_input_size, 1 };
    } else {
        inputShape = { lstm_batch_size, lstm_sequence_len, lstm_input_size, 1 };
    }
    cldnn::tensor hiddenStateShape = { lstm_batch_size, 1, lstm_hidden_size, 1 };
    cldnn::layout inputLayout = cldnn::layout(DataTypeFromPrecision(lstmPrecision), cldnn::format::bfyx, inputShape);
    topology.add(cldnn::reshape(inReshapeID, inputPrimitives[0], inputShape));
    topology.add(cldnn::reorder(permuteID, inReshapeID, inputLayout));

    topology.add(cldnn::reshape(inHiddenReshapeID+"_1", inputPrimitives[1], hiddenStateShape));
    topology.add(cldnn::reshape(inHiddenReshapeID+"_2", inputPrimitives[2], hiddenStateShape));

    addInnerPrimitiveToProfiler(inReshapeID, layerName, layer);
    addInnerPrimitiveToProfiler(permuteID, layerName, layer);
    addInnerPrimitiveToProfiler(inHiddenReshapeID+"_1", layerName, layer);
    addInnerPrimitiveToProfiler(inHiddenReshapeID+"_2", layerName, layer);

    for (int i = 0; i < lstm_sequence_len; ++i)
        input_ids_offsets.push_back({ get_string_id(i), {0, i, 0, 0} });

    cldnn::primitive_id inputSplitID = layerName + "_inputSplit";

    if (permute_input) {
        topology.add(cldnn::permute(layerName + "_inputSwap", permuteID, { 1, 0, 2, 3 }));
        addInnerPrimitiveToProfiler(layerName + "_inputSwap", layerName, layer);
        topology.add(cldnn::split(inputSplitID, layerName + "_inputSwap", input_ids_offsets));
    } else {
        topology.add(cldnn::split(inputSplitID, permuteID, input_ids_offsets));
    }
    addInnerPrimitiveToProfiler(inputSplitID, layerName, layer);

    cldnn::tensor gemmSz = cldnn::tensor{ lstm_batch_size, 1, 4 * lstm_hidden_size, 1 };
    cldnn::layout gemmLayout = cldnn::layout(DataTypeFromPrecision(lstmPrecision), cldnn::format::bfyx, gemmSz);
    cldnn::tensor hiddenSz = cldnn::tensor{ lstm_batch_size, 1, lstm_hidden_size, 1 };
    cldnn::tensor cellCropSz = cldnn::tensor{0, 1, 0, 0};
    std::string hiddenStr = hasInitialHidden ? inHiddenReshapeID+"_1" : "";
    std::string cellStr = hasInitialCell ? inHiddenReshapeID+"_2" : "";

    for (int i = 0; i < lstm_sequence_len; ++i) {
        std::string concatID = layerName + "_inputConcat" + get_string_id(i);
        std::string lstm_fc_id = layerName + "_fully_connected" + get_string_id(i);
        std::string lstm_fc_resh_id = layerName + "_gemmReshape" + get_string_id(i);
        std::string lstm_fc_reor_id = layerName + "_gemmReorder" + get_string_id(i);
        std::string lstm_elt_id = layerName + "_lstm_elt" + get_string_id(i);
        std::string crop_id = layerName + "_crop" + get_string_id(i);

        int seqIdx = isForward ? i : lstm_sequence_len - 1 - i;
        if (hiddenStr != "") {
            topology.add(cldnn::concatenation(concatID, { inputSplitID + ":" + get_string_id(seqIdx), hiddenStr },
                            cldnn::concatenation::concatenation_axis::along_x));
            addInnerPrimitiveToProfiler(concatID, layerName, layer);
            topology.add(cldnn::fully_connected(lstm_fc_id, concatID, weightID, hasBias ? biasID : ""));
            addInnerPrimitiveToProfiler(lstm_fc_id, layerName, layer);
            addInnerPrimitiveToProfiler(inputSplitID + ":" + get_string_id(seqIdx), layerName, layer);
        } else {
            topology.add(cldnn::fully_connected(lstm_fc_id, inputSplitID + ":" + get_string_id(seqIdx), weightID, hasBias ? biasID : ""));
            addInnerPrimitiveToProfiler(lstm_fc_id, layerName, layer);
        }

        topology.add(cldnn::reshape(lstm_fc_resh_id, lstm_fc_id, gemmSz));
        topology.add(cldnn::reorder(lstm_fc_reor_id, lstm_fc_resh_id, gemmLayout));
        topology.add(cldnn::lstm_elt(lstm_elt_id, lstm_fc_reor_id,
                                            cellStr, 0, 0, {}, {},
                                            cldnn::lstm_weights_order::fizo));
        addInnerPrimitiveToProfiler(lstm_fc_resh_id, layerName, layer);
        addInnerPrimitiveToProfiler(lstm_fc_reor_id, layerName, layer);
        addInnerPrimitiveToProfiler(lstm_elt_id, layerName, layer);

        hiddenStr = crop_id + ":hidden";
        cellStr = crop_id + ":cell";
        topology.add(cldnn::crop(hiddenStr, lstm_elt_id, hiddenSz, cldnn::tensor{ 0, 0, 0, 0 }));
        addInnerPrimitiveToProfiler(hiddenStr, layerName, layer);
        output_ids_offsets.push_back(hiddenStr);

        if (i < lstm_sequence_len - 1) {
            topology.add(cldnn::crop(cellStr, lstm_elt_id, hiddenSz, cellCropSz));
            addInnerPrimitiveToProfiler(cellStr, layerName, layer);
        } else {
            // last hidden state crop (output 2)
            if (layer->outData.size() > 1) {
                cldnn::primitive_id outputHiddenID = layer_type_lower(layer) + ":" + layer->outData[1]->getName();
                primitiveIDs[hiddenStr] = hiddenStr;
                primitiveIDs[outputHiddenID] = hiddenStr;
            }

            // last cell state crop (output 3)
            if (layer->outData.size() > 2) {
                topology.add(cldnn::crop(cellStr, lstm_elt_id, hiddenSz, cellCropSz));
                cldnn::primitive_id outputCellID = layer_type_lower(layer) + ":" + layer->outData[2]->getName();
                addInnerPrimitiveToProfiler(cellStr, layerName, layer);
                primitiveIDs[outputCellID] = cellStr;
            }
        }
    }

    if (!isForward) std::reverse(output_ids_offsets.begin(), output_ids_offsets.end());

    if (permute_input) {
        topology.add(cldnn::concatenation(layerName + "_outputConcat", output_ids_offsets, cldnn::concatenation::along_f));
        addInnerPrimitiveToProfiler(layerName + "_outputConcat", layerName, layer);
        topology.add(cldnn::permute(layerName, layerName + "_outputConcat", { 1, 0, 2, 3 }));
    } else {
        topology.add(cldnn::concatenation(layerName, output_ids_offsets, cldnn::concatenation::along_f));
    }
    primitiveIDs[layer_type_lower(layer) + ":" + layer->outData[0]->getName()] = layerName;
    addPrimitiveToProfiler(layerName, layer);
}

void Program::CreateDynamicLSTM(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer) {
    int lstm_batch_size, lstm_sequence_len, lstm_input_size, lstm_hidden_size;
    bool hasBias = false, reverseSeq = false;
    auto inputPrimitives = GetPrevLayersPrimitives(layer);

    auto lstmPrecision = layer->outData[0]->getPrecision();
    auto elementSize = cldnn::data_type_traits::size_of(DataTypeFromPrecision(lstmPrecision));
    std::string layerName = layer_type_name_ID(layer);
    cldnn::primitive_id weightID = layerName + m_weightsTag;
    cldnn::primitive_id recurrentID = weightID + "_recurrent";
    cldnn::primitive_id biasID = layerName + m_biasesTag;
    auto rnnLayer = as<RNNSequenceLayer*>(layer);
    bool permute_input = (1 != rnnLayer->axis);
    int32_t directions = 1;

    /* check incoming CNN layer and setup required variables */
    {
        if (rnnLayer->cellType != RNNSequenceLayer::LSTM)
            THROW_IE_EXCEPTION << "RNN layer supports only LSTM like cell";

        auto in_data0 = layer->insData[0].lock();
        if (!in_data0)
            THROW_IE_EXCEPTION << "Missing first input for RNN layer " << layer->name;

        const auto in_dims0 = in_data0->getTensorDesc().getDims();
        const auto out_dims0 = layer->outData[0]->getTensorDesc().getDims();

        auto in_data1 = layer->insData[1].lock();
        auto in_data2 = layer->insData[2].lock();
        auto in_data3 = layer->insData[3].lock();

        if (in_dims0.size() != 3 ||
            in_data1->getTensorDesc().getDims().size() != 2 ||
            in_data2->getTensorDesc().getDims().size() != 2 ||
            in_data3->getTensorDesc().getDims().size() != 1)
            THROW_IE_EXCEPTION << "Wrong input shapes for dynamic RNN Layer " << layer->name;

        if (!permute_input) {
            lstm_batch_size = in_dims0.front();
            lstm_sequence_len = in_dims0[1];
        } else {
            lstm_batch_size = in_dims0[1];
            lstm_sequence_len = in_dims0.front();
        }

        lstm_input_size = in_dims0.back();
        lstm_hidden_size = out_dims0.back();

        if (rnnLayer->direction == RNNSequenceLayer::BDR) {
            directions = 2;
        } else {
            reverseSeq = rnnLayer->direction == RNNSequenceLayer::BWD;
        }
    }

    /* Prepare weight/bias memory primitives - split weight blob into W and R */
    {
        const size_t WchunkSz = lstm_input_size * elementSize;
        const size_t RchunkSz = lstm_hidden_size * elementSize;

        cldnn::tensor wTensor = cldnn::tensor(cldnn::batch(1), cldnn::feature(directions), cldnn::spatial(lstm_input_size, 4 * lstm_hidden_size));
        cldnn::tensor rTensor = cldnn::tensor(cldnn::batch(1), cldnn::feature(directions), cldnn::spatial(lstm_hidden_size, 4 * lstm_hidden_size));
        cldnn::layout WLayout = cldnn::layout(DataTypeFromPrecision(lstmPrecision), m_defaultFormat, wTensor);
        cldnn::layout RLayout = cldnn::layout(DataTypeFromPrecision(lstmPrecision), m_defaultFormat, rTensor);

        auto wLayer = as<InferenceEngine::WeightableLayer *>(layer);

        {
            auto pWeightsBlob = wLayer->_weights;
            auto blobBytes = static_cast<const char *>(pWeightsBlob->buffer());

            auto wmem = cldnn::memory::allocate(*m_engine, WLayout);
            auto wtmpPointer = wmem.pointer<char>();  // implicitly maps buffer - unmap in destructor

            auto rmem = cldnn::memory::allocate(*m_engine, RLayout);
            auto rtmpPointer = rmem.pointer<char>();

            auto wBytes = wtmpPointer.data();
            auto rBytes = rtmpPointer.data();

            for (int h = 0; h < 4 * lstm_hidden_size; h++) {
                // copy "input size" elements to W
                for (size_t b = 0; b < WchunkSz; b++)
                    *wBytes++ = *blobBytes++;

                // copy "lstm_hidden_size" elements to R
                for (size_t b = 0; b < RchunkSz; b++)
                    *rBytes++ = *blobBytes++;
            }

            topology.add(cldnn::data(weightID, wmem));
            topology.add(cldnn::data(recurrentID, rmem));
        }

        /* create bias memory primitive */
        auto pBiasBlob = wLayer->_biases;
        if (pBiasBlob != nullptr) {
            cldnn::tensor bTensor = cldnn::tensor(cldnn::batch(1), cldnn::feature(directions), cldnn::spatial(4 * lstm_hidden_size, 1));
            cldnn::layout BLayout = cldnn::layout(DataTypeFromPrecision(pBiasBlob->getTensorDesc().getPrecision()), m_defaultFormat, bTensor);

            auto bmem = cldnn::memory::allocate(*m_engine, BLayout);
            auto btmpPointer = bmem.pointer<char>();

            auto blobBytes = static_cast<const char *>(pBiasBlob->buffer());
            const size_t BchunkSz = lstm_hidden_size * elementSize;
            auto bBytes = btmpPointer.data();

            for (size_t b = 0; b < 4 * BchunkSz; b++)
                *bBytes++ = *blobBytes++;

            topology.add(cldnn::data(biasID, bmem));
            hasBias = true;
        }
    }

    cldnn::primitive_id inReshapeID = layerName + "_inReshape";
    cldnn::primitive_id permuteID = layerName + "_inputReorder";
    cldnn::primitive_id inHiddenReshapeID = layerName + "_inHiddenReshape";

    cldnn::tensor inputShape;

    if (permute_input) {
        inputShape = { lstm_sequence_len, lstm_batch_size, lstm_input_size, directions };
    } else {
        inputShape = { lstm_batch_size, lstm_sequence_len, lstm_input_size, directions };
    }
    cldnn::tensor hiddenStateShape = { lstm_batch_size, 1, lstm_hidden_size, directions };
    cldnn::layout inputLayout = cldnn::layout(DataTypeFromPrecision(lstmPrecision), cldnn::format::bfyx, inputShape);
    topology.add(cldnn::reshape(inReshapeID, inputPrimitives[0], inputShape));
    topology.add(cldnn::reorder(permuteID, inReshapeID, inputLayout));

    addInnerPrimitiveToProfiler(inReshapeID, layerName, layer);
    addInnerPrimitiveToProfiler(permuteID, layerName, layer);

    topology.add(cldnn::reshape(inHiddenReshapeID + "_1", inputPrimitives[1], hiddenStateShape));
    topology.add(cldnn::reshape(inHiddenReshapeID + "_2", inputPrimitives[2], hiddenStateShape));

    addInnerPrimitiveToProfiler(inHiddenReshapeID + "_1", layerName, layer);
    addInnerPrimitiveToProfiler(inHiddenReshapeID + "_2", layerName, layer);

    cldnn::primitive_id dynID = layerName + "_dynLength";
    cldnn::primitive_id dynReshapeID = layerName + "_dynReshape";
    cldnn::tensor dynShape = { 1, 1, lstm_batch_size, 1 };
    cldnn::layout dynLayout = cldnn::layout(DataTypeFromPrecision(lstmPrecision), cldnn::format::bfyx, dynShape);
    topology.add(cldnn::reshape(dynReshapeID, inputPrimitives[3], dynShape));
    topology.add(cldnn::reorder(dynID, dynReshapeID, dynLayout));

    addInnerPrimitiveToProfiler(dynReshapeID, layerName, layer);
    addInnerPrimitiveToProfiler(dynID, layerName, layer);

    cldnn::primitive_id inputID = permuteID;
    cldnn::primitive_id prevInputID = permuteID;

    if (permute_input) {
        inputID = layerName + "_inputSwap";
        topology.add(cldnn::permute(inputID, prevInputID, { 1, 0, 2, 3 }));
        prevInputID = inputID;
        addInnerPrimitiveToProfiler(inputID, layerName, layer);
    }

    cldnn::primitive_id seq_len_id = layer->name + "seq_lengths";
    if (reverseSeq) {
        inputID = layerName + "_inputReverse";
        topology.add(cldnn::reverse_sequence(inputID, prevInputID, dynID, 1, 0));
        primitivesToIRLayersMap[inputID] = { layer->name };
        addInnerPrimitiveToProfiler(inputID, layerName, layer);
        prevInputID = inputID;
    }

    // last hidden state crop (output 2)
    cldnn::primitive_id outputHiddenID = "", outputCellID = "";
     if (layer->outData.size() > 1) {
        outputHiddenID = layer_type_lower(layer) + ":" + layer->outData[1]->getName();
        auto last_hidden_mem = cldnn::memory::allocate(*m_engine,
        { DataTypeFromPrecision(lstmPrecision),
            cldnn::format::bfyx, { lstm_batch_size, 1, lstm_hidden_size, directions } });
        topology.add(cldnn::mutable_data(outputHiddenID, last_hidden_mem));
        primitiveIDs[outputHiddenID] = outputHiddenID;
    }

    // last cell state crop (output 3)
    if (layer->outData.size() > 2) {
        outputCellID = layer_type_lower(layer) + ":" + layer->outData[2]->getName();
        auto last_cell_mem = cldnn::memory::allocate(*m_engine,
        { DataTypeFromPrecision(lstmPrecision),
            cldnn::format::bfyx, { lstm_batch_size, 1, lstm_hidden_size, directions } });
        topology.add(cldnn::mutable_data(outputCellID, last_cell_mem));
        primitiveIDs[outputCellID] = outputCellID;
    }

    // main part - dLSTM primitive intself
    cldnn::primitive_id dlstmID = layerName + "_dlstm";
    topology.add(cldnn::lstm_dynamic(dlstmID, inputID, dynID,
        weightID, recurrentID, outputHiddenID, outputCellID, biasID,
        inHiddenReshapeID + "_1", inHiddenReshapeID + "_2"));
    prevInputID = inputID = dlstmID;
    addInnerPrimitiveToProfiler(dlstmID, layerName, layer);

    if (reverseSeq) {
        inputID = layerName + "_outputReverse";
        topology.add(cldnn::reverse_sequence(inputID, prevInputID, dynID, 1, 0));
        addInnerPrimitiveToProfiler(inputID, layerName, layer);
        prevInputID = inputID;
    }

    if (permute_input) {
        inputID = layerName + "_outputSwap";
        topology.add(cldnn::permute(inputID, prevInputID, { 1, 0, 2, 3 }));
        addInnerPrimitiveToProfiler(inputID, layerName, layer);
        prevInputID = inputID;
    }

    primitiveIDs[inputID] = inputID;
    primitiveIDs[layer_type_lower(layer) + ":" + layer->outData[0]->getName()] = inputID;
    addPrimitiveToProfiler(layerName, layer, inputID);
}

void Program::CreateRNNPrimitive(cldnn::topology& topology, InferenceEngine::CNNLayerPtr &layer) {
    if (layer->insData.size() > 3) {
        CreateDynamicLSTM(topology, layer);
    } else {
        CreateRegularLSTM(topology, layer);
    }
}

};  // namespace CLDNNPlugin
