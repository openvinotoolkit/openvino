// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstring>
#include <iostream>
#include <details/ie_exception.hpp>
#include <gna_plugin_log.hpp>
#include <limits>
#include "backend/gna_types.h"
#include "quantization.h"

#ifdef DEBUG
#define QUANTWARNING(...) (fprintf(stderr, __VA_ARGS__))
#else
#define QUANTWARNING(...)
#endif


template<>
void QuantizationCallback<int16_t, int32_t>::runFakeQuantize() const {
    uint32_t num_saturate = 0;

    for (uint32_t row = 0; row < num_rows; row++) {
        for (uint32_t col = 0; col < num_columns; col++) {
            float rounding_value = (ptr_float_weights[row * num_columns + col] > 0) ? 0.5f : -0.5f;
            float value = ptr_float_weights[row * num_columns + col];
            if (!*ptr_quantized_weights) {
                value = value * *ptr_weight_scale_factor + rounding_value;
            } else {
                value -= MAX_VAL_2B_WEIGHT;
            }

            int16_t* ptr_weight_16 = ptr_int_weights + (row * num_columns_padded + col);

            if (*ptr_quantized_weights &&
                (value > std::numeric_limits<int16_t>::max() ||
                    value < std::numeric_limits<int16_t>::min())) {
                THROW_GNA_EXCEPTION << "unsupported weights range for I16 quantisation: " << value;
            }

            if (value > std::numeric_limits<int16_t>::max()) {
                *ptr_weight_16 = std::numeric_limits<int16_t>::max();
                num_saturate++;
            } else if (value < std::numeric_limits<int16_t>::min()) {
                *ptr_weight_16 = std::numeric_limits<int16_t>::min();
                num_saturate++;
            } else {
                *ptr_weight_16 = (int16_t)value;
            }
        }
        for (uint32_t col = num_columns; col < num_columns_padded; col++) {
            int16_t* ptr_weight_16 = ptr_int_weights + (row * num_columns_padded + col);
            *ptr_weight_16 = 0;
        }
    }
    for (uint32_t row = num_rows; row < num_rows_padded; row++) {
        for (uint32_t col = 0; col < num_columns_padded; col++) {
            int16_t* ptr_weight_16 = ptr_int_weights + (row * num_columns_padded + col);
            *ptr_weight_16 = 0;
        }
    }

    // case for element wise layer
    if (ptr_float_biases != nullptr && ptr_int_biases != nullptr) {
        for (uint32_t j = 0; j < num_rows; j++) {
            float rounding_value = (ptr_float_biases[j] > 0) ? 0.5f : -0.5f;
            float value = ptr_float_biases[j] * *ptr_output_scale_factor + rounding_value;
            if (value > 2147483647.0) {
                ptr_int_biases[j] = 2147483647L;
                num_saturate++;
            } else if (value < -2147483648.0) {
                ptr_int_biases[j] = -2147483648LL;
                num_saturate++;
            } else {
                ptr_int_biases[j] = (int32_t)value;
            }
        }
        for (uint32_t j = num_rows; j < num_rows_padded; j++) {
            ptr_int_biases[j] = 0;
        }
    }

    if (num_saturate > 0) {
        QUANTWARNING("Warning:  %d / %d saturations in QuantizeAffine16()\n",
            num_saturate,
            num_rows * num_columns + num_rows);
    }
}

template<>
void QuantizationCallback<int16_t, int32_t>::runQuantize() const {
    uint32_t num_saturate = 0;

    if (*ptr_weight_scale_factor == 1.0) {
        // scale factor for weights is not calculated yet
        float mean_weight = 0.0;
        float mean_weight_squared = 0.0;
        float max_weight = -1e20f;
        float var_weight;
        float mean_plus_2stdev;

        for (uint32_t i = 0; i < num_rows; i++) {
            for (uint32_t j = 0; j < num_columns; j++) {
                float weight = ptr_float_weights[i * num_columns + j];
                mean_weight += weight;
                mean_weight_squared += weight * weight;
                if (fabs(weight) > max_weight) {
                    max_weight = fabs(weight);
                }
            }
        }

        mean_weight /= static_cast<float>(num_rows * num_columns);
        mean_weight_squared /= static_cast<float>(num_rows * num_columns);
        var_weight = mean_weight_squared - mean_weight * mean_weight;
        mean_plus_2stdev = mean_weight + 2.0f * static_cast<float>(sqrtf(var_weight));

        if (max_weight != 0.0f) {
            *ptr_weight_scale_factor = static_cast<float>(MAX_VAL_2B_WEIGHT) / max_weight;
        }
        *ptr_output_scale_factor = input_scale_factor * *ptr_weight_scale_factor;
    }

    for (uint32_t row = 0; row < num_rows; row++) {
        for (uint32_t col = 0; col < num_columns; col++) {
            float rounding_value = (ptr_float_weights[row * num_columns + col] > 0) ? 0.5f : -0.5f;
            float value = ptr_float_weights[row * num_columns + col] * *ptr_weight_scale_factor + rounding_value;
            int16_t *ptr_weight_16 = ptr_int_weights + (row * num_columns_padded + col);
            if (value > 32767.0) {
                *ptr_weight_16 = 32767;
                num_saturate++;
            } else if (value < -32768.0) {
                *ptr_weight_16 = -32768;
                num_saturate++;
            } else {
                *ptr_weight_16 = (int16_t) value;
            }
        }
        for (uint32_t col = num_columns; col < num_columns_padded; col++) {
            int16_t *ptr_weight_16 = ptr_int_weights + (row * num_columns_padded + col);
            *ptr_weight_16 = 0;
        }
    }
    for (uint32_t row = num_rows; row < num_rows_padded; row++) {
        for (uint32_t col = 0; col < num_columns_padded; col++) {
            int16_t *ptr_weight_16 = ptr_int_weights + (row * num_columns_padded + col);
            *ptr_weight_16 = 0;
        }
    }

    // case for element wise layer
    if (ptr_float_biases != nullptr && ptr_int_biases != nullptr) {
        for (uint32_t j = 0; j < num_rows; j++) {
            float rounding_value = (ptr_float_biases[j] > 0) ? 0.5f : -0.5f;
            float value = ptr_float_biases[j] * *ptr_output_scale_factor + rounding_value;
            if (value > 2147483647.0) {
                ptr_int_biases[j] = 2147483647L;
                num_saturate++;
            } else if (value < -2147483648.0) {
                ptr_int_biases[j] = -2147483648LL;
                num_saturate++;
            } else {
                ptr_int_biases[j] = (int32_t) value;
            }
        }
        for (uint32_t j = num_rows; j < num_rows_padded; j++) {
            ptr_int_biases[j] = 0;
        }
    }

    if (num_saturate > 0) {
        QUANTWARNING("Warning:  %d / %d saturations in QuantizeAffine16()\n",
                     num_saturate,
                     num_rows * num_columns + num_rows);
    }
}

float ScaleFactorForQuantization(void *ptr_float_memory, float target_max, size_t num_elements) {
    float *ptr_float_feat = reinterpret_cast<float *>(ptr_float_memory);
    float max = 0.0;
    float scale_factor;

    for (size_t i = 0; i < num_elements; i++) {
        if (fabs(ptr_float_feat[i]) > max) {
            max = fabs(ptr_float_feat[i]);
        }
    }

    if (max == 0) {
        scale_factor = -1.0f;  // need to handle all zeros as a special case
    } else {
        scale_factor = target_max / max;
    }

    return (scale_factor);
}

void QuantizeVector16(float *ptr_float_memory, int16_t *ptr_int_memory, uint32_t num_elements, float scale_factor) {
    float *ptr_float_feat = reinterpret_cast<float *>(ptr_float_memory);
    uint32_t num_saturate = 0;

    int16_t *ptr_int_feat = reinterpret_cast<int16_t *>(ptr_int_memory);
    for (uint32_t i = 0; i < num_elements; i++) {
        float rounding_value = (ptr_float_feat[i] > 0) ? 0.5f : -0.5f;
        float value = ptr_float_feat[i] * scale_factor + rounding_value;
        if (value > 32767.0) {
            ptr_int_feat[i] = 32767;
            num_saturate++;
        } else if (value < -32768.0) {
            ptr_int_feat[i] = -32768;
            num_saturate++;
        } else {
            ptr_int_feat[i] = (int16_t) value;
        }
    }

    if (num_saturate > 0) {
        QUANTWARNING("Warning:  %d / %d saturations during QuantizeVector16()\n", num_saturate, num_elements);
    }
}

template<>
void QuantizationCallback<int8_t, gna_compound_bias_t>::runFakeQuantize() const {
    uint32_t num_saturate = 0;

    if (fq_ptr_output_high == nullptr || fq_ptr_output_low == nullptr) {
        THROW_GNA_EXCEPTION << "Fake quantized output range not set";
    }
    if (fq_levels == 0 || fq_levels == 1) {
        THROW_GNA_EXCEPTION << "Fake quantized levels not set";
    }

    for (uint32_t i = 0; i < num_rows; i++) {
        uint32_t channel_multiplier = ((fq_ptr_output_high[i] - fq_ptr_output_low[i]) *
            *ptr_weight_scale_factor) / (fq_levels - 1) + 0.5f;
        ptr_int_biases[i].multiplier = static_cast<uint8_t> (channel_multiplier);
        if (channel_multiplier > MAX_OUT_MULTIPLIER) {
            THROW_GNA_EXCEPTION << "invalid channel multiplier: " << channel_multiplier;
        }

        for (uint32_t j = 0; j < num_columns; j++) {
            auto offset = i * num_columns + j;
            auto rounding_value = (ptr_float_weights[i * num_columns + j] > 0) ? 0.5f : -0.5f;
            float value = ptr_float_weights[offset];
            if (!*ptr_quantized_weights) {
                value = value * (*ptr_weight_scale_factor / ptr_int_biases[i].multiplier) + rounding_value;
            } else {
                value -= MAX_VAL_1B_WEIGHT;
            }
            auto normalizedWeight = static_cast<int32_t>(value);

            if (*ptr_quantized_weights &&
                (value > std::numeric_limits<int8_t>::max() ||
                value < std::numeric_limits<int8_t>::min())) {
                THROW_GNA_EXCEPTION << "unsupported weights range for I8 quantization: " << value;
            }

            if (value > std::numeric_limits<int8_t>::max()) {
                normalizedWeight = std::numeric_limits<int8_t>::max();
                num_saturate++;
            } else if (value < std::numeric_limits<int8_t>::min()) {
                normalizedWeight = std::numeric_limits<int8_t>::min();
                num_saturate++;
            } else {
                normalizedWeight = (int8_t)value;
            }

            // range checking
            ptr_int_weights[offset] = static_cast<int8_t>(normalizedWeight);
        }

        for (uint32_t j = num_columns; j < num_columns_padded; j++) {
            ptr_int_weights[i * num_columns + j] = 0;
        }
    }

    for (uint32_t i = num_rows; i < num_rows_padded; i++) {
        for (uint32_t j = 0; j < num_columns_padded; j++) {
            ptr_int_weights[i * num_columns + j] = 0;
        }
        ptr_int_biases[i].multiplier = 0;
    }

    if (ptr_float_biases != nullptr) {
        for (uint32_t j = 0; j < num_rows; j++) {
            float rounding_value = (ptr_float_biases[j] > 0) ? 0.5f : -0.5f;
            float value = ptr_float_biases[j] * *ptr_output_scale_factor + rounding_value;
            if (value > 2147483647.0) {
                ptr_int_biases[j].bias = 2147483647L;
                num_saturate++;
            } else if (value < -2147483648.0) {
                ptr_int_biases[j].bias = -2147483648LL;
                num_saturate++;
            } else {
                ptr_int_biases[j].bias = (int32_t) value;
            }
        }
    }
    if (num_saturate > 0) {
        QUANTWARNING("Warning:  %d / %d saturations in QuantizeAffine8()\n", num_saturate, num_rows * num_columns + num_rows);
    }
}

template<>
void QuantizationCallback<int8_t, gna_compound_bias_t>::runQuantize() const {
    if (ptr_int_biases == nullptr) {
        THROW_IE_EXCEPTION << "Int biases are empty";
    }
    uint32_t num_saturate = 0;

    if (*ptr_weight_scale_factor == 1.0) {
        // scale factor for weights is not calculated yet
        float mean_weight = 0.0;
        float mean_weight_squared = 0.0;
        float max_weight = -1e20f;
        float var_weight;
        float mean_plus_2stdev;

        for (uint32_t i = 0; i < num_rows; i++) {
            for (uint32_t j = 0; j < num_columns; j++) {
                float weight = ptr_float_weights[i*num_columns + j];
                mean_weight += weight;
                mean_weight_squared += weight * weight;
                if (fabs(weight) > max_weight) {
                    max_weight = fabs(weight);
                }
            }
        }

        mean_weight /= static_cast<float>(num_rows * num_columns);
        mean_weight_squared /= static_cast<float>(num_rows * num_columns);
        var_weight = mean_weight_squared - mean_weight * mean_weight;
        mean_plus_2stdev = mean_weight + 2.0f * static_cast<float>(sqrtf(var_weight));

        *ptr_weight_scale_factor = static_cast<float>(MAX_VAL_1B_WEIGHT) / max_weight;

        // For 8 bit weights quantize as follows:
        // 1. adjust scale factor to increase dynamic range of entire matrix by max multiplier
        // 2. find maximum scaled weight for each row
        // 3. find multiplier such that dividing by the multiplier brings row back within 8-bit dynamic range
        // 4. quantize and store scaled row
        *ptr_weight_scale_factor = MAX_OUT_MULTIPLIER * *ptr_weight_scale_factor;  //  increase dynamic range by max multiplier
        *ptr_output_scale_factor = input_scale_factor * *ptr_weight_scale_factor;
    }
    float valueAcc = 0.0;
    for (uint32_t row = 0; row < num_rows; row++) {
        float scaled_row_max = 0;
        float rounding_value, value;
        for (uint32_t col = 0; col < num_columns; col++) {
            value = ptr_float_weights[row*num_columns + col] * *ptr_weight_scale_factor;
            valueAcc += value;
            if (fabs(value) > scaled_row_max) {
                scaled_row_max = fabs(value);
            }
        }

        value = scaled_row_max / static_cast<float>(MAX_VAL_1B_WEIGHT);
        ptr_int_biases[row].multiplier = (uint8_t) (value + 0.5);
        for (uint32_t col = 0; col < num_columns; col++) {
            int8_t *ptr_weight_8 = ptr_int_weights + (row * num_columns_padded + col);
            rounding_value = (ptr_float_weights[row * num_columns + col] > 0) ? 0.5f : -0.5f;


            value = ptr_float_weights[row * num_columns + col] * (*ptr_weight_scale_factor / ptr_int_biases[row].multiplier) + rounding_value;
            if (value > 127.0) {
                *ptr_weight_8 = 127;
                num_saturate++;
            } else if (value < -128.0) {
                *ptr_weight_8 = -128;
                num_saturate++;
            } else {
                *ptr_weight_8 = (int8_t) value;
            }
        }
        for (uint32_t col = num_columns; col < num_columns_padded; col++) {
            int8_t *ptr_weight_8 = ptr_int_weights + (row * num_columns_padded + col);
            *ptr_weight_8 = 0;
        }
    }
    for (uint32_t row = num_rows; row < num_rows_padded; row++) {
        for (uint32_t col = 0; col < num_columns_padded; col++) {
            int8_t *ptr_weight_8 = ptr_int_weights + (row*num_columns_padded + col);
            *ptr_weight_8 = 0;
        }
        ptr_int_biases[row].multiplier = 0;
    }

    // bias value of the bas will be only used when input bias provided
    if (ptr_float_biases != nullptr) {
        for (uint32_t j = 0; j < num_rows; j++) {
            float rounding_value = (ptr_float_biases[j] > 0) ? 0.5f : -0.5f;
            float value = ptr_float_biases[j] * *ptr_output_scale_factor + rounding_value;
            if (value > 2147483647.0) {
                ptr_int_biases[j].bias = 2147483647L;
                num_saturate++;
            } else if (value < -2147483648.0) {
                ptr_int_biases[j].bias = -2147483648LL;
                num_saturate++;
            } else {
                ptr_int_biases[j].bias = (int32_t) value;
            }
        }
    }

    if (num_saturate > 0) {
        QUANTWARNING("Warning:  %d / %d saturations in QuantizeAffine8()\n", num_saturate, num_rows * num_columns + num_rows);
    }
}
