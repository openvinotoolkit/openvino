// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cstdint>

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
void FixedQuantizeAffine16(float *ptr_float_weights,
                           float *ptr_float_biases,
                           int16_t *ptr_int_weights,
                           int32_t *ptr_int_biases,
                           float input_scale_factor,
                           float weight_scale_factor,
                           float *ptr_output_scale_factor,
                           uint32_t num_rows,
                           uint32_t num_columns,
                           uint32_t num_rows_padded,
                           uint32_t num_columns_padded);
float ScaleFactorForQuantization(void *ptr_float_memory, float target_max, size_t num_elements);
float ScaleFactorForQuantization(std::vector<std::vector<float>> &input_vectors, float target_max);
float ScaleFactorForQuantization(std::vector<std::vector<float>> &input_vectors,
                                 int index,
                                 int num_group_size,
                                 float target_max);
void QuantizeVector16(float *ptr_float_memory, int16_t *ptr_int_memory, uint32_t num_elements, float scale_factor);
void QuantizeVector16(std::vector<std::vector<float>> &input_vectors,
                      int16_t *ptr_int_memory,
                      uint32_t index,
                      uint32_t num_group_size,
                      float scale_factor);
void ReQuantizeVector16(int16_t *ptr_int_memory, uint32_t num_elements, float prev_scale_factor, float scale_factor);
bool IntegrityCheckAffine16(float *ptr_float_weights,
                            float *ptr_float_biases,
                            int16_t *ptr_int_weights,
                            int32_t *ptr_int_biases,
                            float weight_scale_factor,
                            float output_scale_factor,
                            uint32_t num_rows,
                            uint32_t num_columns,
                            uint32_t num_rows_padded,
                            uint32_t num_columns_padded);
bool IntegrityCheckAffineWeights16(float *ptr_float_weights,
                                   int16_t *ptr_int_weights,
                                   float weight_scale_factor,
                                   uint32_t num_rows,
                                   uint32_t num_columns,
                                   uint32_t num_rows_padded,
                                   uint32_t num_columns_padded);
void QuantizeBias16(float *ptr_float_biases,
                    int32_t *ptr_int_biases,
                    float input_scale_factor,
                    float weight_scale_factor,
                    float *ptr_output_scale_factor,
                    uint32_t num_rows);
void DeQuantizeVector16(int16_t *ptr_int_memory, std::vector<float> &float_vector, float scale_factor);
void DeQuantizeVector32(int32_t *ptr_int_memory, std::vector<float> &float_vector, float scale_factor);
void DeQuantizeVector32(int32_t *ptr_int_memory,
                        std::vector<float> &float_vector,
                        uint32_t index,
                        uint32_t num_group_size,
                        float scale_factor);

#include "gna-api.h"

void QuantizeAffine8(float *ptr_float_weights, float *ptr_float_biases, int8_t *ptr_int_weights, intel_compound_bias_t *ptr_int_biases,
                     float input_scale_factor, float *ptr_weight_scale_factor, float *ptr_output_scale_factor,
                     uint32_t num_rows, uint32_t num_columns, uint32_t num_rows_padded, uint32_t num_columns_padded);
void QuantizeBias8(float *ptr_float_biases, intel_compound_bias_t  *ptr_int_biases, float input_scale_factor,
                   float weight_scale_factor, float *ptr_output_scale_factor, uint32_t num_rows);
bool IntegrityCheckAffine8(float *ptr_float_weights, float *ptr_float_biases, int8_t *ptr_int_weights, intel_compound_bias_t *ptr_int_biases,
                           float weight_scale_factor, float output_scale_factor, uint32_t num_rows, uint32_t num_columns,
                           uint32_t num_rows_padded, uint32_t num_columns_padded);


