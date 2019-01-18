// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#define LSTM_GIFO_X_C (component_index)
#define LSTM_GIFO_R_C (component_index+1)
#define LSTM_INPUT_GATE_C (component_index+2)
#define LSTM_INPUT_SIGMOID_C (component_index+3)
#define LSTM_FORGET_GATE_C (component_index+4)
#define LSTM_FORGET_SIGMOID_C (component_index+5)
#define LSTM_CELL_INPUT1_C (component_index+6)
#define LSTM_CELL_INPUT1_TANH_C (component_index+7)
#define LSTM_CELL_INPUT2_C (component_index+8)
#define LSTM_CELL_OUTPUT1_C (component_index+9)
#define LSTM_CELL_TANH_C (component_index+10)
#define LSTM_CELL_OUTPUT2_C (component_index+11)
#define LSTM_CELL_CLIPPING_C (component_index+12)
#define LSTM_OUTPUT_GATE_C (component_index+13)
#define LSTM_OUTPUT_SIGMOID_C (component_index+14)
#define LSTM_HIDDEN_C (component_index+15)
#define LSTM_HIDDEN_IDENTITY_C (component_index+16)
#define LSTM_PROJECTED_C (component_index+17)
#define LSTM_PROJECTED_IDENTITY_C (component_index+18)
#define NUM_LSTM_COMPONENTS 19

#define BILSTM_GIFO_X_FW_C (component_index)
#define BILSTM_GIFO_R_FW_C (component_index+1)
#define BILSTM_INPUT_GATE_FW_C (component_index+2)
#define BILSTM_INPUT_SIGMOID_FW_C (component_index+3)
#define BILSTM_FORGET_GATE_FW_C (component_index+4)
#define BILSTM_FORGET_SIGMOID_FW_C (component_index+5)
#define BILSTM_CELL_INPUT1_FW_C (component_index+6)
#define BILSTM_CELL_INPUT1_TANH_FW_C (component_index+7)
#define BILSTM_CELL_INPUT2_FW_C (component_index+8)
#define BILSTM_CELL_GATE_FW_C (component_index+9)
#define BILSTM_CELL_OUTPUT1_FW_C (component_index+10)
#define BILSTM_CELL_TANH_FW_C (component_index+11)
#define BILSTM_CELL_COPY_FW_C (component_index+12)
#define BILSTM_OUTPUT_GATE_FW_C (component_index+13)
#define BILSTM_OUTPUT_SIGMOID_FW_C (component_index+14)
#define BILSTM_HIDDEN_FW_C (component_index+15)
#define BILSTM_HIDDEN_IDENTITY_FW_C (component_index+16)
#define BILSTM_GIFO_X_BW_C (component_index+17)
#define BILSTM_GIFO_R_BW_C (component_index+18)
#define BILSTM_INPUT_GATE_BW_C (component_index+19)
#define BILSTM_INPUT_SIGMOID_BW_C (component_index+20)
#define BILSTM_FORGET_GATE_BW_C (component_index+21)
#define BILSTM_FORGET_SIGMOID_BW_C (component_index+22)
#define BILSTM_CELL_INPUT1_BW_C (component_index+23)
#define BILSTM_CELL_INPUT1_TANH_BW_C (component_index+24)
#define BILSTM_CELL_INPUT2_BW_C (component_index+25)
#define BILSTM_CELL_GATE_BW_C (component_index+26)
#define BILSTM_CELL_OUTPUT1_BW_C (component_index+27)
#define BILSTM_CELL_TANH_BW_C (component_index+28)
#define BILSTM_CELL_COPY_BW_C (component_index+29)
#define BILSTM_OUTPUT_GATE_BW_C (component_index+30)
#define BILSTM_OUTPUT_SIGMOID_BW_C (component_index+31)
#define BILSTM_HIDDEN_BW_C (component_index+32)
#define BILSTM_HIDDEN_IDENTITY_BW_C (component_index+33)
#define NUM_BILSTM_COMPONENTS 34

#include "gna-api.h"

#define ACTIVATION_SCALE_IG  1024.0f
#define ACTIVATION_SCALE_CI1 1024.0f
#define ACTIVATION_SCALE_CO1 2048.0f
#define ACTIVATION_SCALE_OG  2048.0f
#define ACTIVATION_SCALE_HID 2048.0f
#define MAX_WEIGHT_IFO_GATE  1024.0f
#define NUM_WEIGHT_BYTES_IN        2
#define NUM_WEIGHT_BYTES_PROJ    2

typedef struct {
    float min;
    float max;
    float sum;
    float sum_squared;
    uint32_t num_saturations;
    uint32_t num_elements;
} intel_buffer_stats_t;

typedef struct {
    intel_nnet_layer_t in;        // combined input transform
    intel_nnet_layer_t rec;        // combined recurrent transform
    intel_nnet_layer_t ig;        // input gate
    intel_nnet_layer_t fg;        // forget gate
    intel_nnet_layer_t ci1;        // cell gate input part 1
    intel_nnet_layer_t ci2;        // cell gate input part 2
    intel_nnet_layer_t co1;        // cell gate output part 1
    intel_nnet_layer_t co2;        // cell gate output part 2
    intel_nnet_layer_t og;        // output gate
    intel_nnet_layer_t hid;        // hidden gated output
    intel_nnet_layer_t proj;    // projected output
} intel_lstm_projected_layer_t;

typedef struct {
    intel_affine_layer_t *in;        // combined input transform
    intel_affine_layer_t *rec;        // combined recurrent transform
    intel_affine_layer_t *ig;        // input gate
    intel_affine_layer_t *fg;        // forget gate
    intel_affine_layer_t *ci1;        // cell gate input part 1
    intel_affine_layer_t *ci2;        // cell gate input part 2
    intel_affine_layer_t *co1;        // cell gate output part 1
    intel_affine_layer_t *co2;        // cell gate output part 2
    intel_affine_layer_t *og;        // output gate
    intel_affine_layer_t *hid;        // hidden gated output
    intel_affine_layer_t *proj;        // projected output
} intel_lstm_projected_transform_t;

typedef struct {
    intel_buffer_stats_t in;        // combined input transform
    intel_buffer_stats_t rec;        // combined recurrent transform
    intel_buffer_stats_t ig;        // input gate
    intel_buffer_stats_t fg;        // forget gate
    intel_buffer_stats_t ci1;        // cell gate input part 1
    intel_buffer_stats_t ci2;        // cell gate input part 2
    intel_buffer_stats_t co1;        // cell gate output part 1
    intel_buffer_stats_t co2;        // cell gate output part 2
    intel_buffer_stats_t og;        // output gate
    intel_buffer_stats_t hid;        // hidden gated output
    intel_buffer_stats_t proj;    // projected output
} intel_lstm_projected_stats_t;

typedef struct {
    intel_nnet_layer_t rec;        // combined recurrent transform
    intel_nnet_layer_t ig;        // input gate
    intel_nnet_layer_t fg;        // forget gate
    intel_nnet_layer_t ci1;        // cell gate input part 1
    intel_nnet_layer_t ci2;        // cell gate input part 2
    intel_nnet_layer_t co1;        // cell gate output part 1
    intel_nnet_layer_t co2;        // cell gate output part 2
    intel_nnet_layer_t og;        // output gate
    intel_nnet_layer_t hid;        // hidden gated output
    intel_nnet_layer_t proj;    // projected output
} intel_lstm_partial_layer_t;

typedef struct {
    intel_affine_layer_t *rec;        // combined recurrent transform
    intel_affine_layer_t *ig;        // input gate
    intel_affine_layer_t *fg;        // forget gate
    intel_affine_layer_t *ci1;        // cell gate input part 1
    intel_affine_layer_t *ci2;        // cell gate input part 2
    intel_affine_layer_t *co1;        // cell gate output part 1
    intel_affine_layer_t *co2;        // cell gate output part 2
    intel_affine_layer_t *og;        // output gate
    intel_affine_layer_t *hid;        // hidden gated output
    intel_affine_layer_t *proj;        // projected output
} intel_lstm_partial_transform_t;

typedef struct {
    intel_buffer_stats_t rec;        // combined recurrent transform
    intel_buffer_stats_t ig;        // input gate
    intel_buffer_stats_t fg;        // forget gate
    intel_buffer_stats_t ci1;        // cell gate input part 1
    intel_buffer_stats_t ci2;        // cell gate input part 2
    intel_buffer_stats_t co1;        // cell gate output part 1
    intel_buffer_stats_t co2;        // cell gate output part 2
    intel_buffer_stats_t og;        // output gate
    intel_buffer_stats_t hid;        // hidden gated output
    intel_buffer_stats_t proj;    // projected output
} intel_lstm_partial_stats_t;

typedef struct {
    intel_nnet_layer_t in;                // combined input transform
    intel_nnet_layer_t dintl;            // interleave x8
    intel_nnet_layer_t intl1;            // deinterleave x2
    intel_nnet_layer_t intl2;            // deinterleave x2
    intel_nnet_layer_t intl3;            // deinterleave x2
    intel_nnet_layer_t intl4;            // deinterleave x2
    intel_lstm_partial_layer_t part[4];    // unrolled part
    intel_nnet_layer_t intl;            // interleave x4
} intel_lstm_projected_layer_g4_t;

typedef struct {
    intel_affine_layer_t *in;                // combined input transform
    intel_lstm_partial_transform_t part[4];  // unrolled part
} intel_lstm_projected_transform_g4_t;

typedef struct {
    intel_buffer_stats_t in;            // combined input transform
    intel_lstm_partial_stats_t part[4];    // unrolled part
} intel_lstm_projected_stats_g4_t;

#define NUM_LSTM_LAYERS 11
#define NUM_LSTM_G4_LAYERS 47

extern const char *intel_lstm_projected_layer_name[NUM_LSTM_LAYERS];
extern const char *intel_lstm_projected_layer_g4_name[NUM_LSTM_G4_LAYERS];
/*
void GetLstmBufferStats(intel_lstm_projected_layer_t *ptr_layer, std::vector<intel_lstm_projected_stats_t> &stats);
void UpdateLstmBufferStats(std::vector<intel_lstm_projected_stats_t> &accum, std::vector<intel_lstm_projected_stats_t> stats);
void ClearLstmBufferStats(std::vector<intel_lstm_projected_stats_t> &stats);
void PrintLstmBufferStats(std::string preamble, std::vector<intel_lstm_projected_stats_t> stats);
uint32_t NumBytesLstmMacroLayer(uint32_t num_inputs, uint32_t num_outputs, uint32_t num_cells, uint32_t num_group_size, uint32_t layer_num, bool is_compact);
void InitLstmMacroLayerG1(intel_lstm_projected_layer_t *ptr_layer, intel_lstm_projected_transform_t *ptr_transform, uint32_t num_inputs, uint32_t num_outputs, uint32_t num_cells);
void InitLstmMacroLayerG4(intel_lstm_projected_layer_g4_t *ptr_layer, intel_lstm_projected_transform_g4_t *ptr_transform, uint32_t num_inputs, uint32_t num_outputs, uint32_t num_cells);
void AllocateLstmMacroLayerG1(intel_lstm_projected_layer_t *ptr_layer, intel_lstm_projected_transform_t *ptr_transform, intel_shared_outputs scratch, uint8_t **ptr_memory, uint32_t *ptr_num_bytes_used, uint32_t num_memory_bytes, bool is_compact);
void AllocateLstmMacroLayerG4(intel_lstm_projected_layer_g4_t *ptr_layer, intel_lstm_projected_transform_g4_t *ptr_transform, intel_shared_outputs scratch, uint8_t **ptr_memory, uint32_t *ptr_num_bytes_used, uint32_t num_memory_bytes, bool is_compact);
void ConnectLstmMacroLayerG1(intel_lstm_projected_layer_t *ptr_layer, intel_lstm_projected_transform_t *ptr_transform);
void ConnectLstmMacroLayerG4(intel_lstm_projected_layer_g4_t *ptr_layer, intel_lstm_projected_transform_g4_t *ptr_transform);
void QuantizeLstmMacroLayerG1(std::vector<intel_dnn_component_t> *ptr_component, uint32_t component_index, intel_lstm_projected_transform_t *ptr_transform, float input_scale, gna_scale_factor_t *scale, uint32_t j);
void QuantizeLstmMacroLayerG4(std::vector<intel_dnn_component_t> *ptr_component, uint32_t component_index, intel_lstm_projected_transform_g4_t *ptr_transform, float input_scale, gna_scale_factor_t *scale, uint32_t j);
void ReQuantizeLstmMacroLayerG1(std::vector<intel_dnn_component_t> *ptr_component, uint32_t component_index, intel_lstm_projected_layer_t *ptr_layer, float input_scale, gna_scale_factor_t *scale, uint32_t j);
void ReQuantizeLstmMacroLayerG4(std::vector<intel_dnn_component_t> *ptr_component, uint32_t component_index, intel_lstm_projected_layer_g4_t *ptr_layer, float input_scale, gna_scale_factor_t *scale, uint32_t j);
void IntegrityCheckLstmMacroLayer(std::vector<intel_dnn_component_t> *ptr_component, uint32_t component_index, intel_lstm_projected_layer_t *ptr_layer, gna_scale_factor_t *scale, uint32_t j);

*/