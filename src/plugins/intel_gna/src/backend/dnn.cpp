// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdio>
#include <cmath>

#include <gna2-model-api.h>
#include "gna2_model_helper.hpp"
#include "gna2_model_debug_log.hpp"

#ifndef _NO_MKL_
#include <mkl_dnn.h>
#endif

#include "runtime/floatmath.h"
#include "dnn.hpp"
#include "gna_plugin_log.hpp"
#include "runtime/pwl.h"
#include "runtime/cnn.h"


void GNAPluginNS::backend::ClearScoreError(intel_score_error_t *error) {
    error->num_scores = 0;
    error->num_errors = 0;
    error->max_error = 0.0;
    error->sum_error = 0.0;
    error->sum_squared_error = 0.0;
    error->max_rel_error = 0.0;
    error->sum_rel_error = 0.0;
    error->sum_squared_rel_error = 0.0;
}

void GNAPluginNS::backend::UpdateScoreError(intel_score_error_t *error, intel_score_error_t *total_error) {
    total_error->num_errors += error->num_errors;
    total_error->num_scores += error->num_scores;
    total_error->sum_error += error->sum_error;
    total_error->sum_squared_error += error->sum_squared_error;
    if (error->max_error > total_error->max_error) {
        total_error->max_error = error->max_error;
    }
    total_error->sum_rel_error += error->sum_rel_error;
    total_error->sum_squared_rel_error += error->sum_squared_rel_error;
    if (error->max_rel_error > total_error->max_rel_error) {
        total_error->max_rel_error = error->max_rel_error;
    }
}

void GNAPluginNS::backend::SoftmaxGoogle(float *ptr_output, float *ptr_input, const uint32_t num_outputs, const uint32_t num_inputs) {
    // Assumes input vector contains log likelihoods
    // The computes x[i] = x[i] - log(sum_j exp(x[j]))
    // This normalizes the likelihoods by the sum of likelihoods but stores them as log likelihoods

    float max_score = ptr_input[0];
    float sum = 0.0;
    float diff;
    // find max score for normalization to [0,1]
    for (uint32_t i = 0; i < num_inputs; i++) {
        if (ptr_input[i] > max_score) {
            max_score = ptr_input[i];
        }
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        sum += exp(ptr_input[i] - max_score);
    }
    if (sum < 1.0e-20) {
        fprintf(stderr, "Warning:  attempt to take log(0) in SoftmaxGoogle()!\n");
        sum = 1.0e-20f;
    }
    diff = max_score + log(sum);
    for (uint32_t i = 0; i < num_outputs; i++) {
        ptr_output[i] = ptr_input[i] - diff;
    }
}
