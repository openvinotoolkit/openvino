// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_lstm_cell.hpp"

TEST_F(myriadLayersTests_nightly, LSTMCellSequenceNet) {
    const size_t input_size = 512;
    const size_t state_size = 128;
    const size_t seq_size = 2;
    const size_t batch_size = 4;

    size_t num_weights = ngates * state_size * (input_size + state_size);
    size_t num_bias = ngates * state_size;

    /* weights generating */
    TBlob<uint8_t>::Ptr weightsBlob_for_net(GenWeights((num_weights + num_bias)));
    ie_fp16 *weights_for_net = static_cast<ie_fp16*>(weightsBlob_for_net->buffer());
    TBlob<uint8_t>::Ptr weightsBlob_tmp(GenWeights((num_weights + num_bias)));
    ie_fp16 *weights0 = static_cast<ie_fp16*>(weightsBlob_tmp->buffer());
    ie_fp16 *weights1 = weights0 + ngates * state_size * input_size;
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
    ie_fp16 *bias = weights0 + num_weights;
    for (int i = 0; i < num_bias; i++) {
        bias[i] = PrecisionUtils::f32tof16((float)((rand() % num_bias)) / num_bias);
        *(weights_for_net + num_weights + i) = bias[i];
    }

    InferenceEngine::Core ie;
    auto full_network = ie.ReadNetwork(tensorIteratorModel, weightsBlob_for_net);
    full_network.addOutput("RNNOutput", 0);

    InferenceEngine::InputsDataMap networkInputsFull;
    networkInputsFull = full_network.getInputsInfo();
    InferenceEngine::OutputsDataMap networkOutputsFull;
    networkOutputsFull = full_network.getOutputsInfo();

    networkInputsFull.begin()->second->setPrecision(InferenceEngine::Precision::FP16);
    (++networkInputsFull.begin())->second->setPrecision(InferenceEngine::Precision::FP16);
    (++++networkInputsFull.begin())->second->setPrecision(InferenceEngine::Precision::FP16);
    networkOutputsFull.begin()->second->setPrecision(InferenceEngine::Precision::FP16);

    InferenceEngine::ExecutableNetwork exeNetworkFull;
    std::map<std::string, std::string> networkConfig;
    ASSERT_NO_THROW(exeNetworkFull = _vpuPluginPtr->LoadNetwork(full_network, networkConfig));
    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = exeNetworkFull.CreateInferRequest());

    InferenceEngine::Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = inferRequest.GetBlob("RNNInput"));

    InferenceEngine::Blob::Ptr inputBlobHidden;
    ASSERT_NO_THROW(inputBlobHidden = inferRequest.GetBlob("RNNInput_Hidden"));
    InferenceEngine::Blob::Ptr inputBlobCellState;
    ASSERT_NO_THROW(inputBlobCellState = inferRequest.GetBlob("RNNInput_CellState"));

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

    /* gates repacking for weights for reference function */
    TBlob<uint8_t>::Ptr weightsBlob_inv_tmp(GenWeights(num_weights + num_bias));
    ie_fp16 *weights_inv0 = static_cast<ie_fp16*>(weightsBlob_inv_tmp->buffer());
    ie_fp16 *weights_inv1 = weights_inv0 + ngates * state_size * input_size;
    ie_fp16 *bias_inv = weights_inv0 + num_weights;
    {
        for (int g = 0; g < ngates; g++)
        {
            int stride = state_size * input_size;
            for (int i = 0; i < stride; i++)
            {
                weights_inv0[g * stride + i] = weights0[gate_map[g] * stride + i];
            }
        }
        for (int g = 0; g < ngates; g++)
        {
            int stride = state_size * state_size;
            for (int i = 0; i < stride; i++)
            {
                weights_inv1[g * stride + i] = weights1[gate_map[g] * stride + i];
            }
        }
        for (int g = 0; g < ngates; g++)
        {
            int stride = state_size;
            for (int i = 0; i < stride; i++)
            {
                bias_inv[g * stride + i] = bias[gate_map[g] * stride + i];
            }
        }
    }
    /* weights repacking */
    auto weightsBlob0_repacked = make_shared_blob<ie_fp16>({Precision::FP16, {1, ngates * state_size * input_size}, Layout::NC});
    weightsBlob0_repacked->allocate();
    auto weightsBlob1_repacked = make_shared_blob<ie_fp16>({Precision::FP16, {1, ngates * state_size * state_size}, Layout::NC});
    weightsBlob1_repacked->allocate();
    ie_fp16* weights0_repacked = static_cast<ie_fp16*>(weightsBlob0_repacked->buffer());
    ie_fp16* weights1_repacked = static_cast<ie_fp16*>(weightsBlob1_repacked->buffer());
    matrix_copy_transpose(weights_inv0, weights0_repacked, ngates * state_size, input_size);
    matrix_copy_transpose(weights_inv1, weights1_repacked, ngates * state_size, state_size);
    /* reference version */
    auto refOut0 = make_shared_blob<ie_fp16>({Precision::FP16, {seq_size * batch_size, state_size}, Layout::NC});
    refOut0->allocate();
    auto refOut1 = make_shared_blob<ie_fp16>({Precision::FP16, {seq_size * batch_size, state_size}, Layout::NC});
    refOut1->allocate();
    auto gatesBlob = make_shared_blob<ie_fp16>({Precision::FP16, {1, ngates * state_size}, Layout::NC});
    gatesBlob->allocate();
    ie_fp16* h_dst = static_cast<ie_fp16*>(refOut0->buffer());
    ie_fp16* c_dst = static_cast<ie_fp16*>(refOut1->buffer());
    ie_fp16* gates = static_cast<ie_fp16*>(gatesBlob->buffer());
    for (size_t b = 0; b < batch_size; b++)
    {
        for (size_t c = 0; c < seq_size; c++)
        {
            lstm_cell(input_size,
                      state_size,
                    // weights
                      weights0_repacked,
                      weights1_repacked,
                      bias_inv,
                    // input
                      src_data + input_size * c + input_size * seq_size * b,
                      (c == 0)?(src_data_hidden + state_size * b):(h_dst + state_size * (c-1) + state_size * seq_size * b),
                      (c == 0)?(src_data_cell_state + state_size * b):(c_dst),
                    // output
                      h_dst + state_size * c + state_size * seq_size * b,
                      c_dst,
                      gates
            );
        }
    }

    ASSERT_NO_THROW(inferRequest.SetBlob("RNNInput_Hidden", inputBlobHidden));
    ASSERT_NO_THROW(inferRequest.SetBlob("RNNInput_CellState", inputBlobCellState));
    ASSERT_NO_THROW(inferRequest.SetBlob("RNNInput", inputBlob));

    ASSERT_NO_THROW(inferRequest.Infer());

    InferenceEngine::Blob::Ptr output;
    ASSERT_NO_THROW(output = inferRequest.GetBlob("RNNOutput"));

    CompareCommonAbsolute(output, refOut0, ERROR_BOUND);
}

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsLSTMCell_smoke,
        ::testing::Values<lstmcell_test_params>(MAKE_STRUCT(lstmcell_test_params, 512, 128))
);
