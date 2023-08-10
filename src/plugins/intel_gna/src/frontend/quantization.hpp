// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include "backend/gna_types.hpp"
#include "quantized_layer_params.hpp"

namespace ov {
namespace intel_gna {
namespace frontend {

#define MAX_OUT_MULTIPLIER 230
#define MAX_VAL_1B_WEIGHT  127
#define MAX_VAL_1B_FEAT    64
#define MAX_VAL_1B_BIAS    127
#define MAX_VAL_2B_WEIGHT  16384
#define MAX_VAL_2B_FEAT    16384
#define MAX_VAL_4B_BIAS    1073741824

// Common data required for quantization of weights and biases
struct QuantizationData {
    const size_t num_rows;
    const size_t num_columns;
    float scale_factor;
    // This field is currently used for weights as well as for biases
    QuantizationParams& weights_quant_params;
};

std::pair<float, float> FindMinMaxValues(void* ptr_float_memory, size_t num_elements);
float ScaleFactorForQuantization(void* ptr_float_memory, float target_max, size_t num_elements);
template <typename T>
extern void QuantizeWeights(const QuantizationData& data,
                            float* ptr_float_weights,
                            T* ptr_int_weights,
                            gna_compound_bias_t* ptr_int_biases,
                            const bool quantized_weights);
template <typename T>
extern void QuantizeBiases(const QuantizationData& data, float* ptr_float_biases, T* ptr_int_biases);

#ifdef __clang__
#    pragma clang diagnostic push
#    ifdef __has_warning
#        if __has_warning("-Wimplicit-const-int-float-conversion")
#            pragma clang diagnostic ignored "-Wimplicit-const-int-float-conversion"
#        elif __has_warning("-Wimplicit-int-float-conversion")
#            pragma clang diagnostic ignored "-Wimplicit-int-float-conversion"
#        endif
#    endif
#endif

template <class T>
inline T SaturationCast(float value, uint32_t* saturation_counter = nullptr) {
    if (value > std::numeric_limits<T>::max()) {
        if (saturation_counter) {
            (*saturation_counter)++;
        }
        return std::numeric_limits<T>::max();
    } else if (value < std::numeric_limits<T>::min()) {
        if (saturation_counter) {
            (*saturation_counter)++;
        }
        return std::numeric_limits<T>::min();
    } else {
        return static_cast<T>(value);
    }
}

#if defined(__clang__)
#    pragma clang diagnostic pop
#endif

/**
 * @brief Apply FQ levels onto a value
 */
float ApplyFQ(float value, float input_low, float input_high, float output_low, float output_high, uint32_t levels);

}  // namespace frontend
}  // namespace intel_gna
}  // namespace ov
