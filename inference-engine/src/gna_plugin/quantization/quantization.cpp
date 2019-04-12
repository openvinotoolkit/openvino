// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstring>
#include <iostream>
#include "quantization.h"

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
                      uint32_t num_columns_padded) {
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

        *ptr_weight_scale_factor = static_cast<float>(MAX_VAL_2B_WEIGHT) / max_weight;
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
                           uint32_t num_columns_padded) {
    uint32_t num_saturate = 0;

    for (uint32_t row = 0; row < num_rows; row++) {
        for (uint32_t col = 0; col < num_columns; col++) {
            float rounding_value = (ptr_float_weights[row * num_columns + col] > 0) ? 0.5f : -0.5f;
            float value = ptr_float_weights[row * num_columns + col] * weight_scale_factor + rounding_value;
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
    }
    for (uint32_t row = num_rows; row < num_rows_padded; row++) {
        for (uint32_t col = 0; col < num_columns_padded; col++) {
            int16_t *ptr_weight_16 = ptr_int_weights + (row * num_columns_padded + col);
            *ptr_weight_16 = 0;
        }
    }

    *ptr_output_scale_factor = input_scale_factor * weight_scale_factor;

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

    if (num_saturate > 0) {
        QUANTWARNING("Warning:  %d / %d saturations in FixedQuantizeAffine16()\n",
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
        scale_factor = 1.0;
    } else {
        scale_factor = target_max / max;
    }

    return (scale_factor);
}

float ScaleFactorForQuantization(std::vector<std::vector<float>> &input_vectors, float target_max) {
    float max = 0.0;
    float scale_factor;
    uint32_t num_vectors = (uint32_t) input_vectors.size();

    for (uint32_t i = 0; i < num_vectors; i++) {
        float *ptr_float_feat = input_vectors[i].data();
        uint32_t num_elements = (uint32_t) input_vectors[i].size();
        for (uint32_t j = 0; i < num_elements; i++) {
            if (fabs(ptr_float_feat[j]) > max) {
                max = fabs(ptr_float_feat[j]);
            }
        }
    }

    if (max == 0) {
        scale_factor = 1.0;
    } else {
        scale_factor = target_max / max;
    }

    return (scale_factor);
}

float ScaleFactorForQuantization(std::vector<std::vector<float>> &input_vectors,
                                 int index,
                                 int num_group_size,
                                 float target_max) {
    float max = 0.0;
    float scale_factor;
    uint32_t start_index = (uint32_t) index;
    uint32_t end_index =
        (uint32_t) ((index + num_group_size > input_vectors.size()) ? input_vectors.size() - 1 : start_index
            + num_group_size);

    for (uint32_t i = start_index; i < end_index; i++) {
        float *ptr_float_feat = input_vectors[i].data();
        uint32_t num_elements = (uint32_t) input_vectors[i].size();
        for (uint32_t j = 0; j < num_elements; j++) {
            if (fabs(ptr_float_feat[j]) > max) {
                max = fabs(ptr_float_feat[j]);
            }
        }
    }

    if (max == 0) {
        scale_factor = 1.0;
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

void QuantizeVector16(std::vector<std::vector<float>> &input_vectors,
                      int16_t *ptr_int_memory,
                      uint32_t index,
                      uint32_t num_group_size,
                      float scale_factor) {
    int16_t *ptr_int_feat = reinterpret_cast<int16_t *> (ptr_int_memory);
    uint32_t num_saturate = 0;
    uint32_t num_elements = (uint32_t) input_vectors[0].size();  // assume all vector are same size
    uint32_t start_index = (uint32_t) index;
    uint32_t end_index =
        (uint32_t) ((index + num_group_size > input_vectors.size()) ? input_vectors.size() - 1 : start_index
            + num_group_size);

    if (end_index - start_index < num_group_size) {
        memset(ptr_int_feat, 0, num_elements * num_group_size * sizeof(int16_t));  // for zero padding partial group
    }
    for (uint32_t j = start_index; j < end_index; j++) {
        for (uint32_t i = 0; i < num_elements; i++) {
            float *ptr_float_feat = input_vectors[j].data();
            float rounding_value = (ptr_float_feat[i] > 0) ? 0.5f : -0.5f;
            float value = ptr_float_feat[i] * scale_factor + rounding_value;
            if (value > 32767.0) {
                ptr_int_feat[i * num_group_size + j - start_index] = 32767;
                num_saturate++;
            } else if (value < -32768.0) {
                ptr_int_feat[i * num_group_size + j - start_index] = -32768;
                num_saturate++;
            } else {
                ptr_int_feat[i * num_group_size + j - start_index] = (int16_t) value;
            }
        }
    }
    if (num_saturate > 0) {
        QUANTWARNING("Warning:  %d / %d saturations during QuantizeVector16()\n",
                     num_saturate,
                     num_elements * num_group_size);
    }
}

void ReQuantizeVector16(int16_t *ptr_int_memory, uint32_t num_elements, float prev_scale_factor, float scale_factor) {
    uint32_t num_saturate = 0;

    int16_t *ptr_int_feat = reinterpret_cast<int16_t *> (ptr_int_memory);
    for (uint32_t i = 0; i < num_elements; i++) {
        float float_value = ptr_int_feat[i] / prev_scale_factor;
        float rounding_value = (float_value > 0) ? 0.5f : -0.5f;
        float value = float_value * scale_factor + rounding_value;
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
        QUANTWARNING("Warning:  %d / %d saturations during ReQuantizeVector16()\n", num_saturate, num_elements);
    }
}

void QuantizeBias16(float *ptr_float_biases,
                    int32_t *ptr_int_biases,
                    float input_scale_factor,
                    float weight_scale_factor,
                    float *ptr_output_scale_factor,
                    uint32_t num_rows) {
    uint32_t num_saturate = 0;

    *ptr_output_scale_factor = input_scale_factor * weight_scale_factor;
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

    if (num_saturate > 0) {
        QUANTWARNING("Warning:  %d / %d saturations in QuantizeBias16()\n", num_saturate, num_rows);
    }
}

void DeQuantizeVector16(int16_t *ptr_int_memory, std::vector<float> &float_vector, float scale_factor) {
    int16_t *int16_vector = reinterpret_cast<int16_t *> (ptr_int_memory);
    for (uint32_t i = 0; i < float_vector.size(); i++) {
        float_vector[i] = int16_vector[i] / scale_factor;
    }
}

void DeQuantizeVector32(int32_t *ptr_int_memory, std::vector<float> &float_vector, float scale_factor) {
    int32_t *int32_vector = reinterpret_cast<int32_t  *> (ptr_int_memory);
    for (uint32_t i = 0; i < float_vector.size(); i++) {
        float_vector[i] = int32_vector[i] / scale_factor;
    }
}

void DeQuantizeVector32(int32_t *ptr_int_memory,
                        std::vector<float> &float_vector,
                        uint32_t index,
                        uint32_t num_group_size,
                        float scale_factor) {
    int32_t *int32_vector = reinterpret_cast<int32_t  *> (ptr_int_memory);
    for (uint32_t i = 0; i < float_vector.size(); i++) {
        float_vector[i] = int32_vector[i * num_group_size + index] / scale_factor;
    }
}
bool IntegrityCheckAffine16(float *ptr_float_weights,
                            float *ptr_float_biases,
                            int16_t *ptr_int_weights,
                            int32_t *ptr_int_biases,
                            float weight_scale_factor,
                            float output_scale_factor,
                            uint32_t num_rows,
                            uint32_t num_columns,
                            uint32_t num_rows_padded,
                            uint32_t num_columns_padded) {
    bool model_ok = true;

    for (uint32_t row = 0; row < num_rows; row++) {
        for (uint32_t col = 0; col < num_columns; col++) {
            float rounding_value = (ptr_float_weights[row * num_columns + col] > 0) ? 0.5f : -0.5f;
            float value = ptr_float_weights[row * num_columns + col] * weight_scale_factor + rounding_value;
            int16_t *ptr_weight_16 = ptr_int_weights + (row * num_columns_padded + col);
            int16_t int_value;
            if (value > 32767.0) {
                int_value = 32767;
            } else if (value < -32768.0) {
                int_value = -32768;
            } else {
                int_value = (int16_t) value;
            }
            if (int_value != *ptr_weight_16) {
                model_ok = false;
            }
        }
        for (uint32_t col = num_columns; col < num_columns_padded; col++) {
            int16_t *ptr_weight_16 = ptr_int_weights + (row * num_columns_padded + col);
            if (*ptr_weight_16 != 0) {
                model_ok = false;
            }
        }
    }
    for (uint32_t row = num_rows; row < num_rows_padded; row++) {
        for (uint32_t col = 0; col < num_columns_padded; col++) {
            int16_t *ptr_weight_16 = ptr_int_weights + (row * num_columns_padded + col);
            if (*ptr_weight_16 != 0) {
                model_ok = false;
            }
        }
    }

    for (uint32_t j = 0; j < num_rows; j++) {
        float rounding_value = (ptr_float_biases[j] > 0) ? 0.5f : -0.5f;
        float value = ptr_float_biases[j] * output_scale_factor + rounding_value;
        int32_t int_value;
        if (value > 2147483647.0) {
            int_value = 2147483647L;
        } else if (value < -2147483648.0) {
            int_value = -2147483648LL;
        } else {
            int_value = (int32_t) value;
        }
        if (int_value != ptr_int_biases[j]) {
            model_ok = false;
        }
    }
    for (uint32_t j = num_rows; j < num_rows_padded; j++) {
        if (ptr_int_biases[j] != 0) {
            model_ok = false;
        }
    }

    return (model_ok);
}

bool IntegrityCheckAffineWeights16(float *ptr_float_weights,
                                   int16_t *ptr_int_weights,
                                   float weight_scale_factor,
                                   uint32_t num_rows,
                                   uint32_t num_columns,
                                   uint32_t num_rows_padded,
                                   uint32_t num_columns_padded) {
    bool model_ok = true;

    for (uint32_t row = 0; row < num_rows; row++) {
        for (uint32_t col = 0; col < num_columns; col++) {
            float rounding_value = (ptr_float_weights[row * num_columns + col] > 0) ? 0.5f : -0.5f;
            float value = ptr_float_weights[row * num_columns + col] * weight_scale_factor + rounding_value;
            int16_t *ptr_weight_16 = ptr_int_weights + (row * num_columns_padded + col);
            int16_t int_value;
            if (value > 32767.0) {
                int_value = 32767;
            } else if (value < -32768.0) {
                int_value = -32768;
            } else {
                int_value = (int16_t) value;
            }
            if (int_value != *ptr_weight_16) {
                model_ok = false;
            }
        }
        for (uint32_t col = num_columns; col < num_columns_padded; col++) {
            int16_t *ptr_weight_16 = ptr_int_weights + (row * num_columns_padded + col);
            if (*ptr_weight_16 != 0) {
                model_ok = false;
            }
        }
    }
    for (uint32_t row = num_rows; row < num_rows_padded; row++) {
        for (uint32_t col = 0; col < num_columns_padded; col++) {
            int16_t *ptr_weight_16 = ptr_int_weights + (row * num_columns_padded + col);
            if (*ptr_weight_16 != 0) {
                model_ok = false;
            }
        }
    }

    return (model_ok);
}


void QuantizeAffine8(float *ptr_float_weights, float *ptr_float_biases,
                     int8_t *ptr_int_weights, intel_compound_bias_t *ptr_int_biases,
                     float input_scale_factor, float *ptr_weight_scale_factor,
                     float *ptr_output_scale_factor, uint32_t num_rows, uint32_t num_columns,
                     uint32_t num_rows_padded, uint32_t num_columns_padded) {
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
            int8_t *ptr_weight_8 = ptr_int_weights + (row*num_columns_padded + col);
            rounding_value = (ptr_float_weights[row * num_columns + col] > 0) ? 0.5f : -0.5f;


            value = ptr_float_weights[row*num_columns + col] * (*ptr_weight_scale_factor / ptr_int_biases[row].multiplier) + rounding_value;
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
            int8_t *ptr_weight_8 = ptr_int_weights + (row*num_columns_padded + col);
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


void QuantizeBias8(float *ptr_float_biases,
                   intel_compound_bias_t  *ptr_int_biases,
                   float input_scale_factor,
                   float weight_scale_factor,
                   float *ptr_output_scale_factor, uint32_t num_rows) {
    uint32_t num_saturate = 0;

    *ptr_output_scale_factor = input_scale_factor * weight_scale_factor;
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
            ptr_int_biases[j].bias = (int32_t)value;
        }
    }

    if (num_saturate > 0) {
        QUANTWARNING("Warning:  %d / %d saturations in QuantizeBias8()\n", num_saturate, num_rows);
    }
}

bool IntegrityCheckAffine8(float *ptr_float_weights, float *ptr_float_biases, int8_t *ptr_int_weights, intel_compound_bias_t *ptr_int_biases,
                           float weight_scale_factor, float output_scale_factor, uint32_t num_rows, uint32_t num_columns,
                           uint32_t num_rows_padded, uint32_t num_columns_padded) {
    bool model_ok = true;

    for (uint32_t row = 0; row < num_rows; row++) {
        float scaled_row_max = 0;
        float rounding_value, value;
        for (uint32_t col = 0; col < num_columns; col++) {
            value = ptr_float_weights[row*num_columns + col] * weight_scale_factor;
            if (fabs(value) > scaled_row_max) {
                scaled_row_max = fabs(value);
            }
        }
        value = scaled_row_max / static_cast<float>(MAX_VAL_1B_WEIGHT);
        if (ptr_int_biases[row].multiplier != (uint8_t)(value + 0.5)) {
            model_ok = false;
        }
        for (uint32_t col = 0; col < num_columns; col++) {
            int8_t *ptr_weight_8 = ptr_int_weights + (row*num_columns_padded + col);
            int8_t int_value;
            rounding_value = (ptr_float_weights[row*num_columns + col] > 0) ? 0.5f : -0.5f;
            value = ptr_float_weights[row*num_columns + col] * (weight_scale_factor / ptr_int_biases[row].multiplier) + rounding_value;
            if (value > 127.0) {
                int_value = 127;
            } else if (value < -128.0) {
                int_value = -128;
            } else {
                int_value = (int8_t)value;
            }
            if (int_value != *ptr_weight_8) {
                model_ok = false;
            }
        }
        for (uint32_t col = num_columns; col < num_columns_padded; col++) {
            int8_t *ptr_weight_8 = ptr_int_weights + (row*num_columns_padded + col);
            if (*ptr_weight_8 != 0) {
                model_ok = false;
            }
        }
    }
    for (uint32_t row = num_rows; row < num_rows_padded; row++) {
        for (uint32_t col = 0; col < num_columns_padded; col++) {
            int8_t *ptr_weight_8 = ptr_int_weights + (row*num_columns_padded + col);
            if (*ptr_weight_8 != 0) {
                model_ok = false;
            }
        }
        if (ptr_int_biases[row].multiplier != 0) {
            model_ok = false;
        }
    }

    for (uint32_t j = 0; j < num_rows; j++) {
        float rounding_value = (ptr_float_biases[j] > 0) ? 0.5f : -0.5f;
        float value = ptr_float_biases[j] * output_scale_factor + rounding_value;
        int32_t int_value;
        if (value > 2147483647.0) {
            int_value = 2147483647L;
        } else if (value < -2147483648.0) {
            int_value = -2147483648LL;
        } else {
            int_value = (int32_t)value;
        }
        if (int_value != ptr_int_biases[j].bias) {
            model_ok = false;
        }
    }

    return(model_ok);
}

