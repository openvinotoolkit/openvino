// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include <cstdint>
#include <cstdio>
#include <gna_plugin_log.hpp>

#include "cnn.h"
#include "backend/dnn_types.h"
#include "backend/gna_limitations.hpp"
#include "gna_lib_ver_selector.hpp"


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
    layer_name = " In layer '" + std::string(component->original_layer_name) + "'";
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

#if GNA_LIB_VER == 2
// a1: fastest changing index
// A - size neede
template <typename T>
T getQubeIndex(T a1, T a2, T a3, T A2, T A3) {
    return a1 * A2 * A3 + a2 * A3 + a3;
}

bool matchesPaddedArea(unsigned filterIndex, unsigned outputIndex, unsigned inputSize, unsigned paddingSize, unsigned stride) {
    const auto paddedIndex = stride * outputIndex + filterIndex;
    if (paddedIndex >= inputSize + 2 * paddingSize) {
        THROW_GNA_EXCEPTION << "In: isZeroPaddingCase, paddedIndex >= inputSize + 2 * paddingSize";
    }
    if (paddedIndex < paddingSize || paddedIndex >= inputSize + paddingSize) {
        return true;
    }
    return false;
}

float CNN2DFilter32SingleHWC(const float bias, const float* filter, const unsigned KH, const unsigned KW, const unsigned KC,
    const float* image, const unsigned IH, const unsigned IW, const unsigned IC,
    const unsigned oh, const unsigned ow, const unsigned oc,
    const std::array<uint32_t, 2>& convStride,
    const std::array<uint32_t, 2>& zeroPadding) {

    const auto cSH = convStride[0];
    const auto cSW = convStride[1];

    const auto zPH = zeroPadding[0];
    const auto zPW = zeroPadding[1];
    float output = 0;
    for (unsigned kh = 0; kh < KH; kh++) {
        for (unsigned kw = 0; kw < KW; kw++) {
            for (unsigned kc = 0; kc < KC; kc++) {
                if (!matchesPaddedArea(kh, oh, IH, zPH, cSH) &&
                    !matchesPaddedArea(kw, ow, IW, zPW, cSW)) {
                    const auto ih = (cSH * oh + kh) - zPH;
                    const auto iw = (cSW * ow + kw) - zPW;
                    const auto ic = kc;
                    const auto imageIndex = getQubeIndex(ih, iw, ic, IW, IC);
                    const auto imageElement = image[imageIndex];
                    const auto filterIndex = getQubeIndex(kh, kw, kc, KW, KC);
                    const auto filterElement = filter[filterIndex];
                    const auto product = imageElement * filterElement;
                    output += product;
                }
            }
        }
    }
    output += bias;
    return output;
}

void CNN2DFilter32(intel_dnn_component_t* component) {
    float* ptr_filters = reinterpret_cast<float*>(component->op.conv2D.ptr_filters);
    float* ptr_biases = reinterpret_cast<float*>(component->op.conv2D.ptr_biases);
    float* ptr_inputs = reinterpret_cast<float*>(component->ptr_inputs);
    float* ptr_outputs = reinterpret_cast<float*>(component->ptr_outputs);

    std::string layer_name;
    layer_name = " In layer '" + std::string(component->original_layer_name) + "'";

    const auto IH = component->tensors[0].dimensions[1]; // NHWC
    const auto IW = component->tensors[0].dimensions[2]; // NHWC
    const auto IC = component->tensors[0].dimensions[3]; // NHWC

    const auto OH = component->tensors[1].dimensions[1]; // NHWC
    const auto OW = component->tensors[1].dimensions[2]; // NHWC
    const auto OC = component->tensors[1].dimensions[3]; // NHWC

    const auto kn = component->tensors[2].dimensions[0]; // NHWC
    const auto kh = component->tensors[2].dimensions[1]; // NHWC
    const auto kw = component->tensors[2].dimensions[2]; // NHWC
    const auto kc = component->tensors[2].dimensions[3]; // NHWC

    if (kn != OC) {
        THROW_GNA_EXCEPTION << "Number of filters should be equal to output depth!" << layer_name;
    }
    if (kc != IC) {
        THROW_GNA_EXCEPTION << "Depth of filter should be equal to input depth!" << layer_name;
    }
    auto kernelIndex = 0;
    for (unsigned oc = 0; oc < OC; oc++) {
        for (unsigned ow = 0; ow < OW; ow++) {
            for (unsigned oh = 0; oh < OH; oh++) {
                const auto outputIndex = getQubeIndex(oh, ow, oc, OW, OC);
                ptr_outputs[outputIndex] = CNN2DFilter32SingleHWC(*(ptr_biases + oc), ptr_filters + kernelIndex, kh, kw, kc,
                    ptr_inputs, IH, IW, IC,
                    oh, ow, oc,
                    component->op.conv2D.convStride,
                    component->op.conv2D.zeroPadding);
            }
        }
        // kernel padded to 16B = 4 * sizeof(float)
        kernelIndex += ALIGN(kh * kw * kc, GNAPluginNS::GNALimitations::convEachKernelByteAlignment / sizeof(float));
    }
}

#endif
