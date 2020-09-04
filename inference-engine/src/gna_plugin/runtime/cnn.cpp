// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include <cstdint>
#include <cstdio>
#include <gna_plugin_log.hpp>

#include "cnn.h"
#include "backend/dnn_types.h"


void CNNFilter32(intel_dnn_component_t *component) {
    float *ptr_filters = reinterpret_cast<float *>(component->op.conv1D.ptr_filters);
    float *ptr_biases = reinterpret_cast<float *>(component->op.conv1D.ptr_biases);
    float *ptr_inputs = reinterpret_cast<float *>(component->ptr_inputs);
    float *ptr_outputs = reinterpret_cast<float *>(component->ptr_outputs);
    uint32_t num_filter_outputs = component->op.conv1D.num_feature_map_rows - component->op.conv1D.num_filter_rows + 1;
    uint32_t
            num_inputs_band_stride = component->op.conv1D.num_feature_maps * component->op.conv1D.num_feature_map_columns;
    uint32_t num_filter_coefficients = component->op.conv1D.num_filter_coefficients;

    std::string layer_name;
#ifdef PLOT
    layer_name = " In layer '" + std::string(component->original_layer_name) + "'";
#endif
    if (component->num_rows_in != 1 || component->num_rows_out != 1) {
        THROW_GNA_EXCEPTION << "Bad number of rows in CNNFilter32!" << layer_name;
    }
    if (component->num_columns_out < num_filter_outputs * component->op.conv1D.num_filters) {
        THROW_GNA_EXCEPTION << "Bad num_columns_out in CNNFilter32!" << layer_name;
    }

    for (uint32_t j = 0; j < num_filter_outputs; j++) {
        float *ptr_in = ptr_inputs + j * num_inputs_band_stride;
        for (uint32_t i = 0; i < component->op.conv1D.num_filters; i++) {
            float *ptr_coef = ptr_filters + i * num_filter_coefficients;
            float sum = ptr_biases[i];
            for (uint32_t k = 0; k < num_filter_coefficients; k++) {
                sum += ptr_in[k] * ptr_coef[k];
            }
            ptr_outputs[j * component->op.conv1D.num_filters + i] = sum;
        }
    }
}

void CNNMaxPool(intel_dnn_component_t *component, intel_dnn_number_type_t number_type) {
    if (number_type == kDnnInt) {
        int32_t *ptr_inputs = reinterpret_cast<int32_t *>(component->ptr_inputs);
        int32_t *ptr_outputs = reinterpret_cast<int32_t *>(component->ptr_outputs);
        uint32_t num_inputs = component->num_columns_in;
        uint32_t num_columns = component->op.maxpool.num_inputs_stride;
        uint32_t num_pool_size = component->op.maxpool.num_inputs;
        uint32_t num_pool_step = component->op.maxpool.num_inputs_step;
        uint32_t num_rows_in = num_inputs / component->op.maxpool.num_inputs_stride;

        for (uint32_t i = 0; i < num_columns; i++) {
            int32_t m = 0;
            if (component->op.maxpool.do_sum_not_max) {
                uint32_t num_saturate = 0;
                for (uint32_t j = 0; j < num_rows_in; j += num_pool_step) {
                    int64_t sum = 0;
                    uint32_t num_end = (j + num_pool_size > num_rows_in) ? num_rows_in : j + num_pool_size;
                    for (uint32_t k = j; k < num_end; k++) {
                        sum += ptr_inputs[k * num_columns + i];
                    }
                    constexpr int32_t sum_max_threshold = std::numeric_limits<int32_t>::max();
                    constexpr int32_t sum_min_threshold = std::numeric_limits<int32_t>::min();
                    if (sum > sum_max_threshold) {
                        ptr_outputs[m * num_columns + i] = sum_max_threshold;
                        num_saturate++;
                    } else if (sum < sum_min_threshold) {
                        ptr_outputs[m * num_columns + i] = sum_min_threshold;
                        num_saturate++;
                    } else {
                        ptr_outputs[m * num_columns + i] = static_cast<int32_t>(sum);
                    }
                    m++;
                }
                if (num_saturate > 0) {
                    fprintf(stderr, "Warning:  %d saturations in CNNMaxPool()\n", num_saturate);
                }
            } else {
                for (uint32_t j = 0; j < num_rows_in; j += num_pool_step) {
                    int32_t max = INT32_MIN;
                    uint32_t num_end = (j + num_pool_size > num_rows_in) ? num_rows_in : j + num_pool_size;
                    for (uint32_t k = j; k < num_end; k++) {
                        if (ptr_inputs[k * num_columns + i] > max) max = ptr_inputs[k * num_columns + i];
                    }
                    ptr_outputs[m * num_columns + i] = max;
                    m++;
                }
            }
        }
    } else {
        float *ptr_inputs = reinterpret_cast<float *>(component->ptr_inputs);
        float *ptr_outputs = reinterpret_cast<float *>(component->ptr_outputs);
        uint32_t num_inputs = component->num_columns_in;
        uint32_t num_columns = component->op.maxpool.num_inputs_stride;
        uint32_t num_pool_size = component->op.maxpool.num_inputs;
        uint32_t num_pool_step = component->op.maxpool.num_inputs_step;
        uint32_t num_rows_in = num_inputs / component->op.maxpool.num_inputs_stride;

        for (uint32_t i = 0; i < num_columns; i++) {
            int32_t m = 0;
            if (component->op.maxpool.do_sum_not_max) {
                for (uint32_t j = 0; j < num_rows_in; j += num_pool_step) {
                    float sum = 0.0;
                    uint32_t num_end = (j + num_pool_size > num_rows_in) ? num_rows_in : j + num_pool_size;
                    for (uint32_t k = j; k < num_end; k++) {
                        sum += ptr_inputs[k * num_columns + i];
                    }
                    ptr_outputs[m * num_columns + i] = sum;
                    m++;
                }
            } else {
                for (uint32_t j = 0; j < num_rows_in; j += num_pool_step) {
                    float max = -1e20f;
                    uint32_t num_end = (j + num_pool_size > num_rows_in) ? num_rows_in : j + num_pool_size;
                    for (uint32_t k = j; k < num_end; k++) {
                        if (ptr_inputs[k * num_columns + i] > max) max = ptr_inputs[k * num_columns + i];
                    }
                    ptr_outputs[m * num_columns + i] = max;
                    m++;
                }
            }
        }
    }
}
