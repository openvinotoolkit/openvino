// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include "backend/gna_types.h"

#define MAX_OUT_MULTIPLIER 230
#define MAX_VAL_1B_WEIGHT 127
#define MAX_VAL_1B_FEAT 64
#define MAX_VAL_1B_BIAS 127
#define MAX_VAL_2B_WEIGHT 16384
#define MAX_VAL_2B_FEAT 16384
#define MAX_VAL_4B_BIAS 1073741824

template <class WeightsType,  class BiasType>
struct QuantizationCallback {
    float *ptr_float_weights;
    float *ptr_float_biases;
    WeightsType* ptr_int_weights;
    BiasType* ptr_int_biases;
    float input_scale_factor;
    float *ptr_weight_scale_factor;
    float *ptr_output_scale_factor;
    uint32_t num_rows;
    uint32_t num_columns;
    uint32_t num_rows_padded;
    uint32_t num_columns_padded;

    bool quantizedWeights;
    size_t fq_levels;
    const size_t fq_num_stats;
    const float *fq_ptr_input_low;
    const float *fq_ptr_input_high;
    const float* fq_ptr_output_low;
    const float* fq_ptr_output_high;

    void runQuantize() const;
    void runFakeQuantize() const;
};

template class QuantizationCallback<int16_t, int32_t>;
template class QuantizationCallback<int8_t, gna_compound_bias_t>;
template class QuantizationCallback<int8_t, int8_t>;

std::pair<float, float> FindMinMaxValues(void* ptr_float_memory, size_t num_elements);
float ScaleFactorForQuantization(void *ptr_float_memory, float target_max, size_t num_elements);
void QuantizeVector16(float *ptr_float_memory, int16_t *ptr_int_memory, uint32_t num_elements, float scale_factor);
