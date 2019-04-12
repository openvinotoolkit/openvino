// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dnn.h"
#include <vector>

#define SIGMOID_NUM_SEGMENTS 65
#define SIGMOID_DOMAIN 10.0f  // portion of input to be approximated (-10,10)
#define TANH_NUM_SEGMENTS 65
#define TANH_DOMAIN 5.0f  // portion of input to be approximated (-5,5)
#define RELU_NUM_SEGMENTS 2
#define LEAKYRELU_SLOPE 0.01
#define IDENTITY_NUM_SEGMENTS 3
#define IDENTITY_DOMAIN 10.0f
#define PWL_MAX_ERR_PERCENT 1.0f
#define PWL_MAX_ITERATIONS 2000
#define PWL_MAX_NUM_SEGMENTS 128
#define PWL_DESIGN_THRESHOLD 0.1f
#define PWL_DESIGN_SAMPLES 500
#define ACTIVATION_SCALE_FACTOR 2048.0f
#define IDENTITY_SCALE_FACTOR 2049.0f
#define XBASEMASK 0xFFFFFFFC  // only top 30 bits are used
#define KALDI_LSTM_CLIP_LOWER (-50.0)
#define KALDI_LSTM_CLIP_UPPER (50.0)

typedef struct {
    double t;
    double alpha;
    double beta;
    double m;
    double b;
} pwl_t;

typedef struct {
    double slope;
    uint64_t slope_scale = 0;
    uint32_t slope_scale_index;
} pwl_gna_slope_scale_t;

double first_deriv_tanh(const double x);
double sigmoid(const double x);
double first_deriv_sigmoid(const double x);
double relu(const double x);
double leaky_relu(const double x);

double clipping(const double x, const double lbound, const double ubound);
void PwlApply16(intel_dnn_component_t *component, const uint32_t num_subset_size);
void PwlApply16(intel_dnn_component_t *component,
                const uint32_t num_row_start,
                const uint32_t num_row_end,
                const uint32_t num_col_start,
                const uint32_t num_col_end);
void PwlApply32(intel_dnn_component_t *component, const uint32_t num_subset_size);
void PwlApply32(intel_dnn_component_t *component,
                const uint32_t num_row_start,
                const uint32_t num_row_end,
                const uint32_t num_col_start,
                const uint32_t num_col_end);
void PwlDesign16(const DnnActivation activation_type,
                 intel_pwl_segment_t *ptr_segment,
                 const uint32_t num_segments,
                 const float scale_in,
                 const float scale_out);
void PwlDesignOpt16(const DnnActivation activation_type,
                std::vector<intel_pwl_segment_t> &ptr_segment,
                const float scale_in,
                const float scale_out);
