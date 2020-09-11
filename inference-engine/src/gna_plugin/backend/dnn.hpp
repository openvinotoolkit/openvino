// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdlib>
#include <cstdio>
#include <memory.h>
#include <xmmintrin.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <type_traits>
#include <vector>

#include "am_intel_dnn.hpp"
#include "dnn_types.h"

#if GNA_LIB_VER == 2
#include <gna2-model-api.h>
#endif

#define DNN_MAX_BATCH_SIZE 8
#define DNN_MAX_INPUTS 3072
#define DNN_MAX_OUTPUTS 8192
#define DNN_MAX_ERROR 1.0e-4f
#define DNN_NUM_BYTES_INT_BIAS 4
#define DNN_NUM_BYTES_INT_AFFINE_OUT 4
#define DNN_RAND_INT8_AMPLITUDE 127.0f
#define DNN_RAND_INT16_AMPLITUDE 16384.0f
#define DNN_RAND_INT32_AMPLITUDE 1048576.0f
#define DNN_RAND_FLOAT32_AMPLITUDE 8.0f

/**
 * whether to dump weights and biases
 */
#define DUMP_WB
/**
 * in light mode only layer names are dumped
 * @param filename
 * @param number_type
 * @return
 */
#define LIGHT_DUMP

namespace GNAPluginNS {
namespace backend {

void ApplyAffineTransform(intel_dnn_component_t *component, uint32_t *list, uint32_t listsize);
void ApplyDiagonalTransform(intel_dnn_component_t *component);
void ApplyRecurrentTransform(intel_dnn_component_t *component, uint32_t row, void *ptr_feedbacks);
void ApplyConvolutional1DTransform(intel_dnn_component_t *component);
void ApplyPiecewiseLinearTransform(intel_dnn_component_t *component,
                                            intel_dnn_number_type_t number_type,
                                            uint32_t listsize);
void ApplyPiecewiseLinearTransform(intel_dnn_component_t *component,
                                            intel_dnn_number_type_t number_type,
                                            uint32_t listsize,
                                            uint32_t num_row);
void ApplyMaxPoolTransform(intel_dnn_component_t *component, intel_dnn_number_type_t number_type);
void ApplyTranspose(intel_dnn_component_t *component);
void ApplyCopy(intel_dnn_component_t *component);

void PlotFloatIntDnn(GNAPluginNS::backend::AMIntelDNN *dnn, GNAPluginNS::backend::AMIntelDNN *dnn_int);
bool isCompatibleDnn(GNAPluginNS::backend::AMIntelDNN dnn1, GNAPluginNS::backend::AMIntelDNN dnn2);
void ClearScoreError(intel_score_error_t *error);
void UpdateScoreError(intel_score_error_t *error, intel_score_error_t *total_error);
void SoftmaxGoogle(float *ptr_output, float *ptr_input, const uint32_t num_outputs, const uint32_t num_inputs);

template <class T>
void AdvanceOperationIfAllApplied(const std::vector<intel_dnn_component_t>& component, int i, T*& operation) {
    if (i == component.size() - 1 || component[i + 1].operation != kDnnPiecewiselinearOp) {
        ++operation;
    }
}

template <class T>
void AdvanceCnnOperationIfAllApplied(const std::vector<intel_dnn_component_t>& component, int i, T*& operation) {
    if (i == component.size() - 1 || ((component[i + 1].operation != kDnnMaxPoolOp)
                                      && (component[i + 1].operation != kDnnPiecewiselinearOp))) {
        operation++;
    }
}

}  // namespace backend
}  // namespace GNAPluginNS
