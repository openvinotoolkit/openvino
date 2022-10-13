// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include "quantized_layer_params.hpp"
#include "backend/gna_types.h"

#define MAX_OUT_MULTIPLIER 230
#define MAX_VAL_1B_WEIGHT 127
#define MAX_VAL_1B_FEAT 64
#define MAX_VAL_1B_BIAS 127
#define MAX_VAL_2B_WEIGHT 16384
#define MAX_VAL_2B_FEAT 16384
#define MAX_VAL_4B_BIAS 1073741824

// Common data required for quantization of weights and biases
struct QuantizationData {
    const size_t num_rows;
    const size_t num_columns;
    float scale_factor;
    // This field is currently used for weights as well as for biases
    GNAPluginNS::QuantizationParams& weights_quant_params;
};

std::pair<float, float> FindMinMaxValues(void* ptr_float_memory, size_t num_elements);
float ScaleFactorForQuantization(void *ptr_float_memory, float target_max, size_t num_elements);
template <typename T>
extern void QuantizeWeights(const QuantizationData& data, float* ptr_float_weights, T* ptr_int_weights,
    gna_compound_bias_t* ptr_int_biases, const bool quantized_weights);
template <typename T>
extern void QuantizeBiases(const QuantizationData& data, float* ptr_float_biases, T* ptr_int_biases);
