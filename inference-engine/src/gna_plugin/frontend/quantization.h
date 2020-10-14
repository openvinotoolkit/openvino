// Copyright (C) 2018-2020 Intel Corporation
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
#define MAX_VAL_2B_WEIGHT 16384
#define MAX_VAL_2B_FEAT 16384
#define MAX_VAL_4B_BIAS 1073741824
#ifdef DEBUG
#define QUANTWARNING(...) (fprintf(stderr, __VA_ARGS__))
#else
#define QUANTWARNING(...)
#endif

void QuantizeAffine16(float *ptr_float_weights,
                      float *ptr_float_biases,
                      int16_t *ptr_int_weights,
                      int32_t *ptr_int_biases,
                      float input_scale_factor,
                      float *ptr_weight_scale_factor,
                      float *ptr_output_scale_factor,
                      uint32_t num_rows,
                      uint32_t num_columns,
                      uint32_t num_rows_padded,
                      uint32_t num_columns_padded);
float ScaleFactorForQuantization(void *ptr_float_memory, float target_max, size_t num_elements);
void QuantizeVector16(float *ptr_float_memory, int16_t *ptr_int_memory, uint32_t num_elements, float scale_factor);
void QuantizeAffine8(float *ptr_float_weights, float *ptr_float_biases, int8_t *ptr_int_weights, gna_compound_bias_t *ptr_int_biases,
                     float input_scale_factor, float *ptr_weight_scale_factor, float *ptr_output_scale_factor,
                     uint32_t num_rows, uint32_t num_columns, uint32_t num_rows_padded, uint32_t num_columns_padded);
