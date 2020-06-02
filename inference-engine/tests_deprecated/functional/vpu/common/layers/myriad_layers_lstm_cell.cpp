// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_lstm_cell.hpp"
#include "blob_factory.hpp"

TEST_F(myriadLayersTests_nightly, LSTMCellSequenceNet) {
    size_t input_size = 2048;
    size_t state_size = 2048;
    size_t seq_size = 16;
    size_t batch_size = 1;
    
    int output_num = 3;

    if (output_num == 1) {
        input_size = 512;
        state_size = 128;
        seq_size = 2;
        batch_size = 4;
    }

    size_t num_weights = ngates * state_size * (input_size + state_size);
    size_t num_bias = ngates * state_size;

    /* weights generating */
    TBlob<uint8_t>::Ptr weightsBlob_for_net(GenWeights((num_weights + num_bias)));
    ie_fp16 *weights_for_net = static_cast<ie_fp16*>(weightsBlob_for_net->buffer());
    TBlob<uint8_t>::Ptr weightsBlob_tmp(GenWeights((num_weights + num_bias)));
    ie_fp16 *weights0 = static_cast<ie_fp16*>(weightsBlob_tmp->buffer());
    ie_fp16 *weights1 = weights0 + ngates * state_size * input_size;
    ie_fp16 *bias = weights0 + num_weights;

    TBlob<uint8_t>::Ptr weightsBlob_inv_tmp(GenWeights(num_weights + num_bias));
    ie_fp16 *weights_inv0 = static_cast<ie_fp16*>(weightsBlob_inv_tmp->buffer());
    ie_fp16 *weights_inv1 = weights_inv0 + ngates * state_size * input_size;
    ie_fp16 *bias_inv = weights_inv0 + num_weights;

    StatusCode st;
    InferenceEngine::IInferRequest::Ptr inferRequest;
    InferenceEngine::Blob::Ptr inputBlob;
    InferenceEngine::Blob::Ptr inputBlobHidden;
    InferenceEngine::Blob::Ptr inputBlobCellState;

    /* reference version */
    auto refOut0 = make_shared_blob<ie_fp16>({Precision::FP16, {seq_size * batch_size, state_size}, Layout::NC});
    refOut0->allocate();
    auto refOut1 = make_shared_blob<ie_fp16>({Precision::FP16, {seq_size * batch_size, state_size}, Layout::NC});
    refOut1->allocate();
    auto refOut2 = make_shared_blob<ie_fp16>({Precision::FP16, {seq_size * batch_size, state_size}, Layout::NC});
    refOut2->allocate();
    auto gatesBlob = make_shared_blob<ie_fp16>({Precision::FP16, {1, ngates * state_size}, Layout::NC});
    gatesBlob->allocate();

    /* weights repacking */
    auto weightsBlob0_repacked = make_shared_blob<ie_fp16>({Precision::FP16, {1, ngates * state_size * input_size}, Layout::NC});
    weightsBlob0_repacked->allocate();
    auto weightsBlob1_repacked = make_shared_blob<ie_fp16>({Precision::FP16, {1, ngates * state_size * state_size}, Layout::NC});
    weightsBlob1_repacked->allocate();
    ie_fp16* weights0_repacked = static_cast<ie_fp16*>(weightsBlob0_repacked->buffer());
    ie_fp16* weights1_repacked = static_cast<ie_fp16*>(weightsBlob1_repacked->buffer());

    ie_fp16* h_dst = static_cast<ie_fp16*>(refOut0->buffer());
    ie_fp16* c_dst = static_cast<ie_fp16*>(refOut1->buffer());
    ie_fp16* l_h_dst = (output_num == 3)? static_cast<ie_fp16*>(refOut2->buffer()) : nullptr;
    ie_fp16* gates = static_cast<ie_fp16*>(gatesBlob->buffer());

    int counter = 0;
    for (int j = 0; j < ngates * state_size; j++) {
        for (int i = 0; i < input_size; i++) {
            weights0[(input_size) * j + i] = PrecisionUtils::f32tof16(((float)(rand() % input_size)) / input_size * 0.01);
            weights_for_net[counter++] = weights0[(input_size) * j + i];
        }
        for (int i = 0; i < state_size; i++) {
            weights1[(state_size) * j + i] = PrecisionUtils::f32tof16(((float)(rand() % state_size)) / state_size * 0.05f);
            weights_for_net[counter++] = weights1[(state_size) * j + i];
        }
    }

    for (int i = 0; i < num_bias; i++) {
        bias[i] = PrecisionUtils::f32tof16((float)((rand() % num_bias)) / num_bias);
        *(weights_for_net + num_weights + i) = bias[i];
    }

    InferenceEngine::Core ie;
    auto full_network = ie.ReadNetwork(tensorIteratorModel, weightsBlob_for_net);
    if (output_num == 3) {
        full_network = ie.ReadNetwork(tensorIteratorModel_2, weightsBlob_for_net);
        full_network.addOutput("lstm_fused_cell/BlockLSTM/TensorIterator", 0);
        full_network.addOutput("lstm_fused_cell/BlockLSTM/TensorIterator", 1);
        full_network.addOutput("lstm_fused_cell/BlockLSTM/TensorIterator", 2);
    } else {
        full_network.addOutput("RNNOutput", 0);
    }

    InferenceEngine::InputsDataMap networkInputsFull;
    networkInputsFull = full_network.getInputsInfo();
    InferenceEngine::OutputsDataMap networkOutputsFull;
    networkOutputsFull = full_network.getOutputsInfo();

    networkInputsFull.begin()->second->setPrecision(InferenceEngine::Precision::FP16);
    (++networkInputsFull.begin())->second->setPrecision(InferenceEngine::Precision::FP16);
    (++++networkInputsFull.begin())->second->setPrecision(InferenceEngine::Precision::FP16);

    networkOutputsFull.begin()->second->setPrecision(InferenceEngine::Precision::FP16);
    if (output_num > 1)
        (++networkOutputsFull.begin())->second->setPrecision(InferenceEngine::Precision::FP16);
    if (output_num > 2)
        (++networkOutputsFull.begin())->second->setPrecision(InferenceEngine::Precision::FP16);

    InferenceEngine::IExecutableNetwork::Ptr exeNetworkFull;
    std::map<std::string, std::string> networkConfig;

    ASSERT_NO_THROW(st = _vpuPluginPtr->LoadNetwork(exeNetworkFull, full_network, networkConfig, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = exeNetworkFull->CreateInferRequest(inferRequest, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = inferRequest->GetBlob("RNNInput", inputBlob, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = inferRequest->GetBlob("RNNInput_Hidden", inputBlobHidden, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = inferRequest->GetBlob("RNNInput_CellState", inputBlobCellState, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    /* input tensors generating */
    ie_fp16 *src_data_cell_state = static_cast<ie_fp16*>(inputBlobCellState->buffer());
    for (int i = 0; i < state_size * batch_size; i++) {
        src_data_cell_state[i] = PrecisionUtils::f32tof16((((float)(rand() % input_size)) / input_size * .1f));
    }
    ie_fp16 *src_data_hidden = static_cast<ie_fp16*>(inputBlobHidden->buffer());
    for (int i = 0; i < state_size * batch_size; i++) {
        src_data_hidden[i] = PrecisionUtils::f32tof16((((float)(rand() % input_size)) / input_size * .1f));
    }
    ie_fp16 *src_data = static_cast<ie_fp16*>(inputBlob->buffer());
    for (int i = 0; i < input_size * batch_size * seq_size; i++) {
        src_data[i] = PrecisionUtils::f32tof16((((float)(rand() % input_size)) / input_size * .1f));
    }
    for (int g = 0; g < ngates; g++) {
        int stride = state_size * input_size;
        for (int i = 0; i < stride; i++) {
            weights_inv0[g * stride + i] = weights0[gate_map[g] * stride + i];
        }
    }
    for (int g = 0; g < ngates; g++) {
        int stride = state_size * state_size;
        for (int i = 0; i < stride; i++) {
            weights_inv1[g * stride + i] = weights1[gate_map[g] * stride + i];
        }
    }
    for (int g = 0; g < ngates; g++) {
        int stride = state_size;
        for (int i = 0; i < stride; i++) {
            bias_inv[g * stride + i] = bias[gate_map[g] * stride + i];
        }
    }

    matrix_copy_transpose(weights_inv0, weights0_repacked, ngates * state_size, input_size);
    matrix_copy_transpose(weights_inv1, weights1_repacked, ngates * state_size, state_size);

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t c = 0; c < seq_size; c++) {
            lstm_cell(input_size,
                    state_size,
                    // weights
                    weights0_repacked,
                    weights1_repacked,
                    bias_inv,
                    // input
                    src_data + input_size * c + input_size * seq_size * b,
                    (c == 0)?(src_data_hidden + state_size * b):(h_dst + state_size * (c - 1) + state_size * seq_size * b),
                    (c == 0)?(src_data_cell_state + state_size * b):(c_dst),

                    output_num,
                    // output
                    h_dst + state_size * c + state_size * seq_size * b,
                    c_dst,
                    (c == seq_size - 1)? l_h_dst + state_size * c + state_size * seq_size * b : nullptr,

                    gates
            );
        }
    }

    ASSERT_NO_THROW(st = inferRequest->SetBlob("RNNInput_Hidden", inputBlobHidden, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    ASSERT_NO_THROW(st = inferRequest->SetBlob("RNNInput_CellState", inputBlobCellState, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    ASSERT_NO_THROW(st = inferRequest->SetBlob("RNNInput", inputBlob, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = inferRequest->Infer(&_resp));

    if (output_num == 1) {
        InferenceEngine::Blob::Ptr output;
        ASSERT_NO_THROW(st = inferRequest->GetBlob("RNNOutput", output, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
        CompareCommonAbsolute(output, refOut0, ERROR_BOUND);
    }
}

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsLSTMCell_smoke,
                        ::testing::Values<lstmcell_test_params>(MAKE_STRUCT(lstmcell_test_params, 512, 128, 3)),
);
