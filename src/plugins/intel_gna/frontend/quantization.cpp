// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstring>
#include <gna_plugin_log.hpp>
#include <limits>
#include "backend/gna_types.h"
#include "quantization.h"
#include <algorithm>

#ifdef DEBUG
#define QUANTWARNING(...) (fprintf(stderr, __VA_ARGS__))
#else
#define QUANTWARNING(...)
#endif


template<>
void QuantizationCallback<int16_t, int32_t>::runFakeQuantize() const {
    if (quantizedWeights) {
        THROW_GNA_EXCEPTION << "Quantized weights are not yet supported in int16 quantization mode";
    }

    uint32_t num_saturate = 0;
    auto input_low = 0.0f;
    auto input_high = 0.0f;
    auto output_low = 0.0f;
    auto output_high = 0.0f;
    size_t levels = 1;
    if (fq_num_stats > 0) {
        input_low = *fq_ptr_input_low;
        input_high = *fq_ptr_input_high;
        output_low = *fq_ptr_output_low;
        output_high = *fq_ptr_output_high;
        levels = fq_levels;
    }

    for (uint32_t row = 0; row < num_rows; row++) {
        for (uint32_t col = 0; col < num_columns; col++) {
            float rounding_value = (ptr_float_weights[row * num_columns + col] > 0) ? 0.5f : -0.5f;
            float value = ptr_float_weights[row * num_columns + col];
            if (fq_num_stats > 0) {
                auto x = value;
                if (x <= std::min(input_low, input_high)) {
                    value = output_low;
                } else if (x > std::max(input_low, input_high)) {
                    value = output_high;
                } else {
                    value = nearbyint((x - input_low) / (input_high - input_low) * (levels - 1)) /
                        (levels - 1) * (output_high - output_low) + output_low;
                }
            }

            value = value * *ptr_weight_scale_factor + rounding_value;

            int16_t* ptr_weight_16 = ptr_int_weights + (row * num_columns_padded + col);

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

std::pair<float, float> FindMinMaxValues(void* ptr_float_memory, size_t num_elements) {
    float* ptr_float_feat = reinterpret_cast<float*>(ptr_float_memory);
    float min = num_elements ? ptr_float_feat[0] : 0.0f;
    float max = num_elements ? ptr_float_feat[0] : 0.0f;

    for (size_t i = 1; i < num_elements; i++) {
        if (fabs(ptr_float_feat[i]) > max) {
            max = fabs(ptr_float_feat[i]);
        }

        if (fabs(ptr_float_feat[i]) < min) {
            min = fabs(ptr_float_feat[i]);
        }
    }

    return { min, max };
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

    auto input_low = 0.0f;
    auto input_high = 0.0f;
    auto output_low = 0.0f;
    auto output_high = 0.0f;
    size_t levels = 1;
    float valueAcc = 0.0f;
    for (uint32_t i = 0; i < num_rows; i++) {
        uint32_t channel_multiplier = 1;
        if (fq_num_stats > 0) {
            auto idx = fq_num_stats == 1 ? 0 : i;
            input_low = fq_ptr_input_low[idx];
            input_high = fq_ptr_input_high[idx];
            output_low = fq_ptr_output_low[idx];
            output_high = fq_ptr_output_high[idx];
            levels = fq_levels;

            channel_multiplier = static_cast<uint32_t>(((input_high - input_low) * *ptr_weight_scale_factor) / (levels - 1));
        } else {
            float scaled_row_max = 0;
            for (uint32_t col = 0; col < num_columns; col++) {
                float value = ptr_float_weights[i * num_columns + col] * *ptr_weight_scale_factor;
                valueAcc += value;
                if (fabs(value) > scaled_row_max) {
                    scaled_row_max = fabs(value);
                }
            }

            channel_multiplier = static_cast<uint32_t>(scaled_row_max / static_cast<float>(MAX_VAL_1B_WEIGHT));
        }

        // channel multiplier shouldn't be 0
        channel_multiplier = channel_multiplier == 0 ? 1 : channel_multiplier;

        ptr_int_biases[i].multiplier = static_cast<uint8_t> (channel_multiplier + 0.5f);
        if (channel_multiplier > MAX_OUT_MULTIPLIER) {
            THROW_GNA_EXCEPTION << "invalid channel multiplier: " << channel_multiplier;
        }

        for (uint32_t j = 0; j < num_columns; j++) {
            auto offset = i * num_columns + j;
            auto rounding_value = (ptr_float_weights[i * num_columns + j] > 0) ? 0.5f : -0.5f;
            float value = ptr_float_weights[offset];
            if (!quantizedWeights) {
                if (fq_num_stats > 0) {
                    auto x = value;
                    if (x <= std::min(input_low, input_high)) {
                        value = output_low;
                    } else if (x > std::max(input_low, input_high)) {
                        value = output_high;
                    } else {
                        value = nearbyint((x - input_low) / (input_high - input_low) * (levels - 1)) /
                            (levels - 1) * (output_high - output_low) + output_low;
                    }
                }

                value = value * (*ptr_weight_scale_factor / ptr_int_biases[i].multiplier) + rounding_value;
            } else {
                value -= MAX_VAL_1B_WEIGHT;
            }
            auto normalizedWeight = static_cast<int32_t>(value);

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
        IE_THROW() << "Int biases are empty";
    }
    uint32_t num_saturate = 0;

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

template<>
void QuantizationCallback<int8_t, int8_t>::runQuantize() const {
    uint32_t num_saturate = 0;
    for (uint32_t row = 0; row < num_rows; row++) {
        for (uint32_t col = 0; col < num_columns; col++) {
            float rounding_value = (ptr_float_weights[row * num_columns + col] > 0) ? 0.5f : -0.5f;
            float value = ptr_float_weights[row * num_columns + col] * *ptr_weight_scale_factor + rounding_value;
            int8_t* ptr_weight_8 = ptr_int_weights + (row * num_columns_padded + col);
            if (value > 127.0) {
                *ptr_weight_8 = 127;
                num_saturate++;
            } else if (value < -128.0) {
                *ptr_weight_8 = -128;
                num_saturate++;
            } else {
                *ptr_weight_8 = (int8_t)value;
            }
        }
        for (uint32_t col = num_columns; col < num_columns_padded; col++) {
            int8_t* ptr_weight_8 = ptr_int_weights + (row * num_columns_padded + col);
            *ptr_weight_8 = 0;
        }
    }
    for (uint32_t row = num_rows; row < num_rows_padded; row++) {
        for (uint32_t col = 0; col < num_columns_padded; col++) {
            int8_t* ptr_weight_8 = ptr_int_weights + (row * num_columns_padded + col);
            *ptr_weight_8 = 0;
        }
    }

    if (ptr_float_biases != nullptr && ptr_int_biases != nullptr) {
        for (uint32_t j = 0; j < num_rows; j++) {
            float rounding_value = (ptr_float_biases[j] > 0) ? 0.5f : -0.5f;
            float value = ptr_float_biases[j] * *ptr_output_scale_factor + rounding_value;
            if (value > 127.0) {
                ptr_int_biases[j] = 127;
                num_saturate++;
            } else if (value < -128.0) {
                ptr_int_biases[j] = -128;
                num_saturate++;
            } else {
                ptr_int_biases[j] = (int8_t)value;
            }
        }
        for (uint32_t j = num_rows; j < num_rows_padded; j++) {
            ptr_int_biases[j] = 0;
        }
    }

    if (num_saturate > 0) {
        QUANTWARNING("Warning:  %d / %d saturations in QuantizeAffine8_8()\n", num_saturate, num_rows * num_columns + num_rows);
    }
}
