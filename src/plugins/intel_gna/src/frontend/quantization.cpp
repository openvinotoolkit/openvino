// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "quantization.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

#include "backend/gna_types.hpp"
#include "layers/gna_fake_quantize_layer.hpp"
#include "log/debug.hpp"
#include "log/log.hpp"

namespace ov {
namespace intel_gna {
namespace frontend {

float ApplyFQ(float value, float input_low, float input_high, float output_low, float output_high, uint32_t levels) {
    if (value <= std::min(input_low, input_high)) {
        return output_low;
    } else if (value > std::max(input_low, input_high)) {
        return output_high;
    } else {
        return nearbyint((value - input_low) / (input_high - input_low) * (levels - 1)) / (levels - 1) *
                   (output_high - output_low) +
               output_low;
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

    return {min, max};
}

float ScaleFactorForQuantization(void* ptr_float_memory, float target_max, size_t num_elements) {
    float* ptr_float_feat = reinterpret_cast<float*>(ptr_float_memory);
    float max = 0.0f;
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

template <typename T>
void QuantizeWeights(const QuantizationData& data,
                     float* ptr_float_weights,
                     T* ptr_int_weights,
                     gna_compound_bias_t* ptr_int_biases,
                     const bool quantized_weights) {}

template <typename T>
void QuantizeBiases(const QuantizationData& data, float* ptr_float_biases, T* ptr_int_biases) {}

template <>
void QuantizeWeights<int8_t>(const QuantizationData& data,
                             float* ptr_float_weights,
                             int8_t* ptr_int_weights,
                             gna_compound_bias_t* ptr_int_biases,
                             const bool quantized_weights) {
    if (!ptr_int_biases && quantized_weights) {
        THROW_GNA_EXCEPTION << "Quantized weights are not yet supported in int8 quantization mode";
    }

    uint32_t num_saturate = 0;
    auto input_low = 0.0f;
    auto input_high = 0.0f;
    auto output_low = 0.0f;
    auto output_high = 0.0f;
    uint32_t levels = 1;
    const auto min_values_size = data.weights_quant_params.GetMinValues().size();

    if (min_values_size > 0) {
        input_low = data.weights_quant_params.GetMinValues(true).front();
        input_high = data.weights_quant_params.GetMaxValues(true).front();
        output_low = data.weights_quant_params.GetMinValues(false).front();
        output_high = data.weights_quant_params.GetMaxValues(false).front();
        levels = static_cast<uint32_t>(data.weights_quant_params.GetLevels());
    }

    for (size_t row = 0; row < data.num_rows; row++) {
        uint32_t channel_multiplier = 1;

        if (ptr_int_biases) {
            if (min_values_size > 0) {
                auto idx = min_values_size == 1 ? 0 : row;
                input_low = data.weights_quant_params.GetMinValues(true).at(idx);
                input_high = data.weights_quant_params.GetMaxValues(true).at(idx);
                output_low = data.weights_quant_params.GetMinValues(false).at(idx);
                output_high = data.weights_quant_params.GetMaxValues(false).at(idx);
                levels = static_cast<uint32_t>(data.weights_quant_params.GetLevels());
                channel_multiplier =
                    static_cast<uint32_t>(((input_high - input_low) * data.scale_factor) / (levels - 1));
            } else {
                float scaled_row_max = 0;
                for (size_t col = 0; col < data.num_columns; col++) {
                    float value = ptr_float_weights[row * data.num_columns + col] * data.scale_factor;
                    if (fabs(value) > scaled_row_max) {
                        scaled_row_max = fabs(value);
                    }
                }

                channel_multiplier =
                    static_cast<uint32_t>((scaled_row_max / static_cast<float>(MAX_VAL_1B_WEIGHT) + 0.5f));
            }

            if (channel_multiplier > MAX_OUT_MULTIPLIER) {
                THROW_GNA_EXCEPTION << "invalid channel multiplier: " << channel_multiplier;
            }

            // channel multiplier shouldn't be 0
            ptr_int_biases[row].multiplier = (channel_multiplier == 0) ? 1 : channel_multiplier;
        }

        for (uint32_t col = 0; col < data.num_columns; col++) {
            auto offset = row * data.num_columns + col;
            float rounding_value = (ptr_float_weights[offset] > 0) ? 0.5f : -0.5f;
            float value = ptr_float_weights[offset];

            if (min_values_size > 0) {
                value = ApplyFQ(value, input_low, input_high, output_low, output_high, levels);
            }

            if (ptr_int_biases) {
                if (quantized_weights) {
                    value -= MAX_VAL_1B_WEIGHT;
                } else {
                    value = value * (data.scale_factor / ptr_int_biases[row].multiplier) + rounding_value;
                }
            } else {
                value = value * data.scale_factor + rounding_value;
            }

            int8_t* ptr_weight_8 = ptr_int_weights + offset;

            *ptr_weight_8 = SaturationCast<int8_t>(value, &num_saturate);
        }
    }

    if (num_saturate > 0) {
        log::warning() << num_saturate << " / " << (data.num_rows * data.num_columns)
                       << " saturations in int8 weights quantization." << std::endl;
    }
}

template <>
void QuantizeWeights<int16_t>(const QuantizationData& data,
                              float* ptr_float_weights,
                              int16_t* ptr_int_weights,
                              gna_compound_bias_t* ptr_int_biases,
                              const bool quantized_weights) {
    if (quantized_weights) {
        THROW_GNA_EXCEPTION << "Quantized weights are not yet supported in int16 quantization mode";
    }

    uint32_t num_saturate = 0;
    auto input_low = 0.0f;
    auto input_high = 0.0f;
    auto output_low = 0.0f;
    auto output_high = 0.0f;
    uint32_t levels = 1;
    const auto min_values_size = data.weights_quant_params.GetMinValues().size();

    if (min_values_size > 0) {
        input_low = data.weights_quant_params.GetMinValues(true).front();
        input_high = data.weights_quant_params.GetMaxValues(true).front();
        output_low = data.weights_quant_params.GetMinValues(false).front();
        output_high = data.weights_quant_params.GetMaxValues(false).front();
        levels = static_cast<uint32_t>(data.weights_quant_params.GetLevels());
    }

    for (size_t row = 0; row < data.num_rows; row++) {
        for (size_t col = 0; col < data.num_columns; col++) {
            float rounding_value = (ptr_float_weights[row * data.num_columns + col] > 0) ? 0.5f : -0.5f;
            float value = ptr_float_weights[row * data.num_columns + col];
            if (min_values_size > 0) {
                value = ApplyFQ(value, input_low, input_high, output_low, output_high, levels);
            }

            value = value * data.scale_factor + rounding_value;

            int16_t* ptr_weight_16 = ptr_int_weights + (row * data.num_columns + col);

            *ptr_weight_16 = SaturationCast<int16_t>(value, &num_saturate);
        }
    }

    if (num_saturate > 0) {
        log::warning() << num_saturate << " / " << (data.num_rows * data.num_columns)
                       << " saturations in int16 weights quantization." << std::endl;
    }
}

template <>
void QuantizeBiases<int8_t>(const QuantizationData& data, float* ptr_float_biases, int8_t* ptr_int_biases) {
    // Stub
}

template <>
void QuantizeBiases<int16_t>(const QuantizationData& data, float* ptr_float_biases, int16_t* ptr_int_biases) {
    // Stub
}

template <>
void QuantizeBiases<int32_t>(const QuantizationData& data, float* ptr_float_biases, int32_t* ptr_int_biases) {
    uint32_t num_saturate = 0;

    // case for element wise layer
    if (ptr_float_biases != nullptr && ptr_int_biases != nullptr) {
        for (size_t row = 0; row < data.num_rows; row++) {
            float rounding_value = (ptr_float_biases[row] > 0) ? 0.5f : -0.5f;
            float value = ptr_float_biases[row] * data.scale_factor + rounding_value;

            ptr_int_biases[row] = SaturationCast<int32_t>(value, &num_saturate);
        }
    }

    if (num_saturate > 0) {
        log::warning() << num_saturate << " / " << data.num_rows << " saturations in int32 biases quantization."
                       << std::endl;
    }
}

template <>
void QuantizeBiases<gna_compound_bias_t>(const QuantizationData& data,
                                         float* ptr_float_biases,
                                         gna_compound_bias_t* ptr_int_biases) {
    uint32_t num_saturate = 0;

    if (ptr_float_biases != nullptr) {
        for (size_t row = 0; row < data.num_rows; row++) {
            float rounding_value = (ptr_float_biases[row] > 0) ? 0.5f : -0.5f;
            float value = ptr_float_biases[row] * data.scale_factor + rounding_value;

            ptr_int_biases[row].bias = SaturationCast<int32_t>(value, &num_saturate);
        }
    }
    if (num_saturate > 0) {
        log::warning() << num_saturate << " / " << data.num_rows << " saturations in compound biases quantization."
                       << std::endl;
    }
}

}  // namespace frontend
}  // namespace intel_gna
}  // namespace ov
