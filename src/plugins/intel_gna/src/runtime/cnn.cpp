// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cnn.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "backend/dnn_types.hpp"
#include "backend/gna_limitations.hpp"
#include "frontend/quantization.hpp"
#include "gna_lib_ver_selector.hpp"
#include "layers/gna_convolution_layer.hpp"
#include "log/debug.hpp"

using namespace ov::intel_gna::gna_convolution_layer;
using namespace ov::intel_gna::limitations;

void CNNFilter32(intel_dnn_component_t* component) {
    auto filters = reinterpret_cast<float*>(component->op.conv1D.ptr_filters);
    auto biases = reinterpret_cast<float*>(component->op.conv1D.ptr_biases);
    auto input = reinterpret_cast<float*>(component->ptr_inputs);
    auto output = reinterpret_cast<float*>(component->ptr_outputs);

    const auto convolutionStride = component->op.conv1D.convStride;
    const auto filterSize = component->op.conv1D.num_filter_coefficients;
    const auto numberOfInputs = component->num_columns_in;
    const auto numberOfOutputsPerFilter = outputFromConv(numberOfInputs, filterSize, convolutionStride);
    const auto numberOfFilters = component->op.conv1D.num_filters;

    std::string layer_name;
    layer_name = " In layer '" + std::string(component->original_layer_name) + "'";
    if (component->num_rows_in != 1 || component->num_rows_out != 1) {
        THROW_GNA_EXCEPTION << "Bad number of rows in CNNFilter32!" << layer_name;
    }
    if (component->num_columns_out < numberOfOutputsPerFilter * numberOfFilters) {
        THROW_GNA_EXCEPTION << "Bad num_columns_out in CNNFilter32!" << layer_name;
    }

    for (uint32_t j = 0; j < numberOfOutputsPerFilter; j++, input += convolutionStride, output += numberOfFilters) {
        auto filter = filters;
        for (uint32_t i = 0; i < numberOfFilters; i++, filter += filterSize) {
            output[i] = biases[i];
            for (uint32_t k = 0; k < filterSize; k++) {
                output[i] += input[k] * filter[k];
            }
        }
    }
}

namespace {

void CNNMaxPoolLegacy(intel_dnn_component_t* component,
                      intel_dnn_number_type_t number_type,
                      const bool sumPoolingOverRide) {
    const uint32_t num_inputs =
        component->op.maxpool.inCHW[0] * component->op.maxpool.inCHW[1] * component->op.maxpool.inCHW[2];
    const uint32_t in_c = component->op.maxpool.inCHW[0];
    const uint32_t num_pool_size = component->op.maxpool.poolingWindowXY[0];
    const uint32_t num_pool_step = component->op.maxpool.poolingStrideXY[0];
    const uint32_t num_rows_in = num_inputs / in_c;

    if (number_type == kDnnInt) {
        int32_t* ptr_inputs = reinterpret_cast<int32_t*>(component->ptr_inputs);
        int32_t* ptr_outputs = reinterpret_cast<int32_t*>(component->ptr_outputs);

        for (uint32_t i = 0; i < in_c; i++) {
            int32_t m = 0;
            if (sumPoolingOverRide) {
                uint32_t num_saturate = 0;
                for (uint32_t j = 0; j < num_rows_in; j += num_pool_step) {
                    int64_t sum = 0;
                    uint32_t num_end = (j + num_pool_size > num_rows_in) ? num_rows_in : j + num_pool_size;
                    for (uint32_t k = j; k < num_end; k++) {
                        sum += ptr_inputs[k * in_c + i];
                    }

                    ptr_outputs[m * in_c + i] =
                        ov::intel_gna::frontend::SaturationCast<int32_t>(static_cast<float>(sum), &num_saturate);
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
                        if (ptr_inputs[k * in_c + i] > max)
                            max = ptr_inputs[k * in_c + i];
                    }
                    ptr_outputs[m * in_c + i] = max;
                    m++;
                }
            }
        }
    } else {
        float* ptr_inputs = reinterpret_cast<float*>(component->ptr_inputs);
        float* ptr_outputs = reinterpret_cast<float*>(component->ptr_outputs);

        for (uint32_t i = 0; i < in_c; i++) {
            int32_t m = 0;
            if (sumPoolingOverRide) {
                for (uint32_t j = 0; j < num_rows_in; j += num_pool_step) {
                    float sum = 0.0;
                    uint32_t num_end = (j + num_pool_size > num_rows_in) ? num_rows_in : j + num_pool_size;
                    for (uint32_t k = j; k < num_end; k++) {
                        sum += ptr_inputs[k * in_c + i];
                    }
                    ptr_outputs[m * in_c + i] = sum;
                    m++;
                }
            } else {
                for (uint32_t j = 0; j < num_rows_in; j += num_pool_step) {
                    float max = std::numeric_limits<float>::lowest();
                    uint32_t num_end = (j + num_pool_size > num_rows_in) ? num_rows_in : j + num_pool_size;
                    for (uint32_t k = j; k < num_end; k++) {
                        if (ptr_inputs[k * in_c + i] > max)
                            max = ptr_inputs[k * in_c + i];
                    }
                    ptr_outputs[m * in_c + i] = max;
                    m++;
                }
            }
        }
    }
}

// a1: fastest changing index
// A - size neede
template <typename T>
T getQubeIndex(T a1, T a2, T a3, T A2, T A3) {
    return a1 * A2 * A3 + a2 * A3 + a3;
}

float MaxPool2D32SingleHWC(const unsigned poolWinH,
                           const unsigned poolWinW,
                           const float* input,
                           const unsigned IH,
                           const unsigned IW,
                           const unsigned IC,
                           const unsigned oh,
                           const unsigned ow,
                           const unsigned oc,
                           const uint32_t poolStrideH,
                           const uint32_t poolStrideW) {
    float output = std::numeric_limits<float>::lowest();
    const auto winStartH = oh * poolStrideH;
    const auto winStartW = ow * poolStrideW;
    for (unsigned winIdxH = 0; winIdxH < poolWinH && winStartH + winIdxH < IH; winIdxH++) {
        for (unsigned winIdxW = 0; winIdxW < poolWinW && winStartW + winIdxW < IW; winIdxW++) {
            const auto inputIndex = getQubeIndex(winStartH + winIdxH, winStartW + winIdxW, oc, IW, IC);
            output = (std::max)(output, input[inputIndex]);
        }
    }
    return output;
}

void CNNMaxPool2DFloat(intel_dnn_component_t* component) {
    float* ptr_inputs = reinterpret_cast<float*>(component->ptr_inputs);
    float* ptr_outputs = reinterpret_cast<float*>(component->ptr_outputs);
    const auto OC = component->op.maxpool.outCHW[0];
    const auto OH = component->op.maxpool.outCHW[1];
    const auto OW = component->op.maxpool.outCHW[2];

    const auto IC = component->op.maxpool.inCHW[0];
    const auto IH = component->op.maxpool.inCHW[1];
    const auto IW = component->op.maxpool.inCHW[2];

    const auto poolWinW = component->op.maxpool.poolingWindowXY[0];
    const auto poolWinH = component->op.maxpool.poolingWindowXY[1];
    const auto poolStrideW = component->op.maxpool.poolingStrideXY[0];
    const auto poolStrideH = component->op.maxpool.poolingStrideXY[1];

    for (unsigned oc = 0; oc < OC; oc++) {
        for (unsigned ow = 0; ow < OW; ow++) {
            for (unsigned oh = 0; oh < OH; oh++) {
                const auto outputIndex = getQubeIndex(oh, ow, oc, OW, OC);
                ptr_outputs[outputIndex] = MaxPool2D32SingleHWC(poolWinH,
                                                                poolWinW,
                                                                ptr_inputs,
                                                                IH,
                                                                IW,
                                                                IC,
                                                                oh,
                                                                ow,
                                                                oc,
                                                                poolStrideH,
                                                                poolStrideW);
            }
        }
    }
}

}  // namespace

namespace {

bool matchesPaddedArea(unsigned filterIndex,
                       unsigned outputIndex,
                       unsigned inputSize,
                       unsigned paddingSize,
                       unsigned stride) {
    const auto paddedIndex = stride * outputIndex + filterIndex;
    if (paddedIndex >= inputSize + 2 * paddingSize) {
        THROW_GNA_EXCEPTION << "In: isZeroPaddingCase, paddedIndex >= inputSize + 2 * paddingSize";
    }
    if (paddedIndex < paddingSize || paddedIndex >= inputSize + paddingSize) {
        return true;
    }
    return false;
}

float CNN2DFilter32SingleHWC(const float bias,
                             const float* filter,
                             const unsigned KH,
                             const unsigned KW,
                             const unsigned KC,
                             const float* image,
                             const unsigned IH,
                             const unsigned IW,
                             const unsigned IC,
                             const unsigned oh,
                             const unsigned ow,
                             const unsigned oc,
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
                if (!matchesPaddedArea(kh, oh, IH, zPH, cSH) && !matchesPaddedArea(kw, ow, IW, zPW, cSW)) {
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

}  // namespace

void CNN2DFilter32(intel_dnn_component_t* component) {
    float* ptr_filters = reinterpret_cast<float*>(component->op.conv2D.ptr_filters);
    float* ptr_biases = reinterpret_cast<float*>(component->op.conv2D.ptr_biases);
    float* ptr_inputs = reinterpret_cast<float*>(component->ptr_inputs);
    float* ptr_outputs = reinterpret_cast<float*>(component->ptr_outputs);

    std::string layer_name;
    layer_name = " In layer '" + std::string(component->original_layer_name) + "'";

    const auto IH = component->tensors[0].dimensions[1];  // NHWC
    const auto IW = component->tensors[0].dimensions[2];  // NHWC
    const auto IC = component->tensors[0].dimensions[3];  // NHWC

    const auto OH = component->tensors[1].dimensions[1];  // NHWC
    const auto OW = component->tensors[1].dimensions[2];  // NHWC
    const auto OC = component->tensors[1].dimensions[3];  // NHWC

    const auto kn = component->tensors[2].dimensions[0];  // NHWC
    const auto kh = component->tensors[2].dimensions[1];  // NHWC
    const auto kw = component->tensors[2].dimensions[2];  // NHWC
    const auto kc = component->tensors[2].dimensions[3];  // NHWC

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
                ptr_outputs[outputIndex] = CNN2DFilter32SingleHWC(*(ptr_biases + oc),
                                                                  ptr_filters + kernelIndex,
                                                                  kh,
                                                                  kw,
                                                                  kc,
                                                                  ptr_inputs,
                                                                  IH,
                                                                  IW,
                                                                  IC,
                                                                  oh,
                                                                  ow,
                                                                  oc,
                                                                  component->op.conv2D.convStride,
                                                                  component->op.conv2D.zeroPadding);
            }
        }
        // kernel padded to 16B = 4 * sizeof(float)
        kernelIndex += ALIGN(kh * kw * kc, Limitations::kConvEachKernelByteAlignment / sizeof(float));
    }
}

namespace {
template <class T>
bool is2D(T&& vec) {
    return vec.size() >= 2 && vec[0] > 1 && vec[1] > 1;
}
}  // namespace

void CNNMaxPool(intel_dnn_component_t* component,
                intel_dnn_number_type_t number_type,
                const bool fused_with_convolution_2d,
                const bool sumPoolingOverRide) {
    if (fused_with_convolution_2d || is2D(component->op.maxpool.poolingStrideXY) ||
        is2D(component->op.maxpool.poolingWindowXY)) {
        if (!sumPoolingOverRide) {
            CNNMaxPool2DFloat(component);
        } else {
            THROW_GNA_EXCEPTION << "SUM pooling2D not supported";
        }
    } else {
        CNNMaxPoolLegacy(component, number_type, sumPoolingOverRide);
    }
}
