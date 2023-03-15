// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_system_conf.h>
#include <stdint.h>

#include "common/numerical_utils.hpp"
#include "preprocessing.hpp"

#if defined(HAVE_AVX2)
#    include <immintrin.h>
#endif  // HAVE_AVX2

namespace ov {
namespace intel_gna {
namespace preprocessing {
#if defined(HAVE_AVX2)
template <typename T>
void quantize_input_avx2(T* dst,
                         const float* src,
                         const uint32_t num_rows,
                         const uint32_t num_columns,
                         const float scale_factor,
                         bool transpose) {
    const uint32_t num_elements = num_rows * num_columns;
    uint32_t moves = num_elements / 8;
    uint32_t mod = num_elements % 8;
    uint32_t i, j;
    uint32_t index = 0;

    __m256 v, zero, half, neg_half, scaleFactors, mask, roundingValues, min, max, values;

    zero = _mm256_setzero_ps();
    half = _mm256_set1_ps(0.5f);
    neg_half = _mm256_set1_ps(-0.5f);
    scaleFactors = _mm256_set1_ps(scale_factor);
    if (std::is_same<T, int8_t>::value) {
        max = _mm256_set1_ps(127.0f);
        min = _mm256_set1_ps(-128.0f);
    } else if (std::is_same<T, int16_t>::value) {
        max = _mm256_set1_ps(32767.0f);
        min = _mm256_set1_ps(-32768.0f);
    }

    for (i = 0; i < moves; i++) {
        v = _mm256_load_ps(&src[i * 8]);

        // roundingValues = (v>0) ? 0.5f : -0.5f;
        mask = _mm256_cmp_ps(v, zero, _CMP_LT_OQ);
        roundingValues = _mm256_blendv_ps(half, neg_half, mask);

        // values = v * scaleFactors +  roundingValues
        values = _mm256_fmadd_ps(v, scaleFactors, roundingValues);

        // shrink to <-32768.0f, 32767.0f>
        values = _mm256_min_ps(values, max);
        values = _mm256_max_ps(values, min);

        // cast
        if (transpose) {
            for (j = 0; j < 8; j++, index++) {
                uint32_t targetColumn = index / num_columns;
                uint32_t targetRow = index % num_columns;
                // target number of rows == source number of columns
                uint32_t targetIndex = targetRow * num_rows + targetColumn;
                dst[targetIndex] = static_cast<T>(values.m256_f32[j]);
            }
        } else {
            for (j = 0; j < 8; j++)
                dst[i * 8 + j] = static_cast<T>(values.m256_f32[j]);
        }
    }

    for (i = 0; i < mod; i++, index++) {
        if (transpose) {
            uint32_t targetColumn = index / num_columns;
            uint32_t targetRow = index % num_columns;
            // target number of rows == source number of columns
            uint32_t targetIndex = targetRow * num_rows + targetColumn;
            if (std::is_same<T, int8_t>::value)
                dst[targetIndex] = common::FloatToInt8WithClamp(src[moves * 8 + i] * scale_factor);
            else if (std::is_same<T, int16_t>::value)
                dst[targetIndex] = common::FloatToInt16WithClamp(src[moves * 8 + i] * scale_factor);
        } else {
            if (std::is_same<T, int8_t>::value)
                dst[moves * 8 + i] = common::FloatToInt8WithClamp(src[moves * 8 + i] * scale_factor);
            else if (std::is_same<T, int16_t>::value)
                dst[moves * 8 + i] = common::FloatToInt16WithClamp(src[moves * 8 + i] * scale_factor);
        }
    }
}

template <typename T>
void dequantize_output_avx2(float* dst,
                            const T* src,
                            uint32_t num_rows,
                            uint32_t num_columns,
                            float scale_factor,
                            bool scale,
                            bool transpose) {
    const uint32_t num_elements = num_rows * num_columns;
    uint32_t moves = num_elements / 8;
    uint32_t mod = num_elements % 8;
    uint32_t i, j;
    uint32_t index = 0;

    __m256 v, scaleFactors, values;
    scaleFactors = _mm256_set1_ps(scale_factor);
    for (i = 0; i < moves; i++) {
        if (std::is_same<T, int8_t>::value) {
            __m256i int32Values = _mm256_cvtepi8_epi32(*reinterpret_cast<const __m128i*>(&src[i * 8]));
            v = _mm256_cvtepi32_ps(int32Values);
        } else if (std::is_same<T, int16_t>::value) {
            __m256i int32Values = _mm256_cvtepi16_epi32(*reinterpret_cast<const __m128i*>(&src[i * 8]));
            v = _mm256_cvtepi32_ps(int32Values);
        } else if (std::is_same<T, int32_t>::value) {
            v = _mm256_cvtepi32_ps(*reinterpret_cast<const __m256i*>(&src[i * 8]));
        }

        // values = v * 1/scaleFactors
        if (scale) {
            values = _mm256_div_ps(v, scaleFactors);
        } else {
            values = v;
        }
        if (transpose) {
            for (j = 0; j < 8; j++, index++) {
                uint32_t targetColumn = index / num_columns;
                uint32_t targetRow = index % num_columns;
                // target number of rows == source number of columns
                uint32_t targetIndex = targetRow * num_rows + targetColumn;
                dst[targetIndex] = values.m256_f32[j];
            }
        } else {
            _mm256_store_ps(&dst[i * 8], values);
        }
    }
    for (i = 0; i < mod; i++, index++) {
        if (transpose) {
            uint32_t targetColumn = index / num_columns;
            uint32_t targetRow = index % num_columns;
            // target number of rows == source number of columns
            uint32_t targetIndex = targetRow * num_rows + targetColumn;
            if (scale) {
                dst[targetIndex] = static_cast<float>(src[moves * 8 + i] / scale_factor);
            } else {
                dst[targetIndex] = static_cast<float>(src[moves * 8 + i]);
            }
        } else {
            if (scale) {
                dst[moves * 8 + i] = static_cast<float>(src[moves * 8 + i] / scale_factor);
            } else {
                dst[moves * 8 + i] = static_cast<float>(src[moves * 8 + i]);
            }
        }
    }
}
#else
template <typename T>
void quantize_input_avx2(T* dst,
                         const float* src,
                         const uint32_t num_rows,
                         const uint32_t num_columns,
                         const float scale_factor,
                         bool transpose) {}

template <typename T>
void dequantize_output_avx2(float* dst,
                            const T* src,
                            uint32_t num_rows,
                            uint32_t num_columns,
                            float scale_factor,
                            bool scale,
                            bool transpose) {}

#endif  // HAVE_AVX2
}  // namespace preprocessing
}  // namespace intel_gna
}  // namespace ov