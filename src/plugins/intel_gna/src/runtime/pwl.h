// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <cstdint>
#include <ngraph/node.hpp>

#include "backend/dnn_types.h"
#include "backend/gna_types.h"

#define SIGMOID_NUM_SEGMENTS 65
#define SIGMOID_DOMAIN 10.0f // portion of input to be approximated (-10,10)
#define TANH_NUM_SEGMENTS 65
#define TANH_DOMAIN 5.0f  // portion of input to be approximated (-5,5)
#define SOFTSIGN_NUM_SEGMENTS 65
#define SOFTSIGN_DOMAIN 10.0f  // portion of input to be approximated (-10,10)
#define RELU_NUM_SEGMENTS 2
#define LEAKYRELU_SLOPE 0.01
#define IDENTITY_NUM_SEGMENTS 3
#define IDENTITY_DOMAIN 10.0f
#define ACTIVATION_SCALE_FACTOR 2048.0f
#define IDENTITY_SCALE_FACTOR 2049.0f
#define XBASEMASK 0xFFFFFFFC  // only top 30 bits are used
#define XBASE_SCALE_INDEX_MASK ~XBASEMASK // only 2 LSB are used
#define KALDI_LSTM_CLIP_LOWER (-50.0)
#define KALDI_LSTM_CLIP_UPPER (50.0)
#define POW_DOMAIN (16.0)
#define POW_NUM_SEGMENTS 65

typedef struct {
    double t;
    double alpha;
    double beta;
    double m;
    double b;
} pwl_t;

double relu(const double x);
double leaky_relu(const double x);

double clipping(const double x, const double lbound, const double ubound);

double pivot_search(std::vector<pwl_t>& result, double(*f)(const double),
                    double(*first_deriv_f)(const double),
                    const uint32_t N,
                    const double alpha_0,
                    const double alpha_N,
                    const double threshold,
                    const bool negative,
                    size_t iter_num);

inline std::vector<pwl_t> negative_pwl(const std::vector<pwl_t>& pwl);

std::vector<pwl_t> pwl_search(const DnnActivation& activation_type,
                              const double l_bound,
                              const double u_bound,
                              const double threshold,
                              const double allowed_err_pct,
                              const int samples,
                              double& err_pct);

bool split_search(const DnnActivationType fun,
                  const double l_bound,
                  const double u_bound);

double calculate_error_pct(const DnnActivation& activation_type,
                           const double l_bound,
                           const double u_bound,
                           const double offset,
                           const int samples);

void PwlApply32(intel_dnn_component_t *component, const uint32_t num_subset_size);
void PwlApply32(intel_dnn_component_t *component,
                const uint32_t num_row_start,
                const uint32_t num_row_end,
                const uint32_t num_col_start,
                const uint32_t num_col_end);
void PwlDesign(const DnnActivation& activation_type,
                 gna_pwl_segment_t *ptr_segment,
                 const uint32_t num_segments,
                 const float scale_in,
                 const float scale_out,
                 const bool low_precision);
void PwlDesignOpt(const DnnActivation& activation_type,
                  const float scale_in,
                  const float scale_out,
                  const bool low_precision,
                  const std::shared_ptr<ngraph::Node>& node,
                  const bool is_fused_with_conv2d,
                  std::vector<gna_pwl_segment_t>& ptr_segment);
