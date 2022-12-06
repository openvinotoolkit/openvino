// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "avx2.hpp"

#if defined(HAVE_AVX2)

#    include <immintrin.h>

#    include "preprocessing.hpp"

void GNAPluginNS::ConvertMatrixFp32ToInt16(int16_t* ptr_dst,
                                           const float* ptr_src,
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
    max = _mm256_set1_ps(32767.0f);
    min = _mm256_set1_ps(-32768.0f);

    for (i = 0; i < moves; i++) {
        v = _mm256_load_ps(&ptr_src[i * 8]);

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
                ptr_dst[targetIndex] = static_cast<int16_t>(values.m256_f32[j]);
            }
        } else {
            for (j = 0; j < 8; j++)
                ptr_dst[i * 8 + j] = static_cast<int16_t>(values.m256_f32[j]);
        }
    }

    for (i = 0; i < mod; i++, index++) {
        if (transpose) {
            uint32_t targetColumn = index / num_columns;
            uint32_t targetRow = index % num_columns;
            // target number of rows == source number of columns
            uint32_t targetIndex = targetRow * num_rows + targetColumn;
            ptr_dst[targetIndex] = ConvertFloatToInt16(ptr_src[moves * 8 + i] * scale_factor);
        } else {
            ptr_dst[moves * 8 + i] = ConvertFloatToInt16(ptr_src[moves * 8 + i] * scale_factor);
        }
    }
}

void GNAPluginNS::ConvertMatrixFp32ToInt8(int8_t* ptr_dst,
                                          const float* ptr_src,
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
    max = _mm256_set1_ps(127.0f);
    min = _mm256_set1_ps(-128.0f);

    for (i = 0; i < moves; i++) {
        v = _mm256_load_ps(&ptr_src[i * 8]);

        // roundingValues = (v>0) ? 0.5f : -0.5f;
        mask = _mm256_cmp_ps(v, zero, _CMP_LT_OQ);
        roundingValues = _mm256_blendv_ps(half, neg_half, mask);

        // values = v * scaleFactors +  roundingValues
        values = _mm256_fmadd_ps(v, scaleFactors, roundingValues);

        // shrink to <-128.0f, 127.0f>
        values = _mm256_min_ps(values, max);
        values = _mm256_max_ps(values, min);

        // cast
        if (transpose) {
            for (j = 0; j < 8; j++, index++) {
                uint32_t targetColumn = index / num_columns;
                uint32_t targetRow = index % num_columns;
                // target number of rows == source number of columns
                uint32_t targetIndex = targetRow * num_rows + targetColumn;
                ptr_dst[targetIndex] = static_cast<int8_t>(values.m256_f32[j]);
            }
        } else {
            for (j = 0; j < 8; j++) {
                ptr_dst[i * 8 + j] = static_cast<int8_t>(values.m256_f32[j]);
            }
        }
    }

    for (i = 0; i < mod; i++, index++) {
        if (transpose) {
            uint32_t targetColumn = index / num_columns;
            uint32_t targetRow = index % num_columns;
            // target number of rows == source number of columns
            uint32_t targetIndex = targetRow * num_rows + targetColumn;
            ptr_dst[targetIndex] = ConvertFloatToInt8(ptr_src[moves * 8 + i] * scale_factor);
        } else {
            ptr_dst[moves * 8 + i] = ConvertFloatToInt8(ptr_src[moves * 8 + i] * scale_factor);
        }
    }
}

void GNAPluginNS::ConvertMatrixInt32ToFp32Avx(float* ptr_dst,
                                              const int32_t* ptr_src,
                                              uint32_t num_rows,
                                              uint32_t num_columns,
                                              float scale_factor,
                                              bool transpose) {
    const uint32_t num_elements = num_rows * num_columns;
    uint32_t moves = num_elements / 8;
    uint32_t mod = num_elements % 8;
    uint32_t i, j;
    uint32_t index = 0;

    __m256 v, scaleFactors, values;
    scaleFactors = _mm256_set1_ps(scale_factor);
    for (i = 0; i < moves; i++) {
        v = _mm256_cvtepi32_ps(*reinterpret_cast<const __m256i*>(&ptr_src[i * 8]));

        // values = v * 1/scaleFactors
        values = _mm256_div_ps(v, scaleFactors);
        if (transpose) {
            for (j = 0; j < 8; j++, index++) {
                uint32_t targetColumn = index / num_columns;
                uint32_t targetRow = index % num_columns;
                // target number of rows == source number of columns
                uint32_t targetIndex = targetRow * num_rows + targetColumn;
                ptr_dst[targetIndex] = values.m256_f32[j];
            }
        } else {
            _mm256_store_ps(&ptr_dst[i * 8], values);
        }
    }
    for (i = 0; i < mod; i++, index++) {
        if (transpose) {
            uint32_t targetColumn = index / num_columns;
            uint32_t targetRow = index % num_columns;
            // target number of rows == source number of columns
            uint32_t targetIndex = targetRow * num_rows + targetColumn;
            ptr_dst[targetIndex] = static_cast<float>(ptr_src[moves * 8 + i] / scale_factor);
        } else {
            ptr_dst[moves * 8 + i] = static_cast<float>(ptr_src[moves * 8 + i] / scale_factor);
        }
    }
}

void GNAPluginNS::ConvertMatrixInt16ToFp32Avx(float* ptr_dst,
                                              const int16_t* ptr_src,
                                              uint32_t num_rows,
                                              uint32_t num_columns,
                                              float scale_factor,
                                              bool transpose) {
    const uint32_t num_elements = num_rows * num_columns;
    uint32_t moves = num_elements / 8;
    uint32_t mod = num_elements % 8;
    uint32_t i, j;
    uint32_t index = 0;

    __m256 v, scaleFactors, values;
    scaleFactors = _mm256_set1_ps(scale_factor);
    for (i = 0; i < moves; i++) {
        __m256i int32Values = _mm256_cvtepi16_epi32(*reinterpret_cast<const __m128i*>(&ptr_src[i * 8]));
        v = _mm256_cvtepi32_ps(int32Values);

        // values = v * 1/scaleFactors
        values = _mm256_div_ps(v, scaleFactors);
        if (transpose) {
            for (j = 0; j < 8; j++, index++) {
                uint32_t targetColumn = index / num_columns;
                uint32_t targetRow = index % num_columns;
                // target number of rows == source number of columns
                uint32_t targetIndex = targetRow * num_rows + targetColumn;
                ptr_dst[targetIndex] = values.m256_f32[j];
            }
        } else {
            _mm256_store_ps(&ptr_dst[i * 8], values);
        }
    }
    for (i = 0; i < mod; i++, index++) {
        if (transpose) {
            uint32_t targetColumn = index / num_columns;
            uint32_t targetRow = index % num_columns;
            // target number of rows == source number of columns
            uint32_t targetIndex = targetRow * num_rows + targetColumn;
            ptr_dst[targetIndex] = static_cast<float>(ptr_src[moves * 8 + i] / scale_factor);
        } else {
            ptr_dst[moves * 8 + i] = static_cast<float>(ptr_src[moves * 8 + i] / scale_factor);
        }
    }
}

void GNAPluginNS::ConvertMatrixInt8ToFp32Avx(float* ptr_dst,
                                             const int8_t* ptr_src,
                                             uint32_t num_rows,
                                             uint32_t num_columns,
                                             float scale_factor,
                                             bool transpose) {
    const uint32_t num_elements = num_rows * num_columns;
    uint32_t moves = num_elements / 8;
    uint32_t mod = num_elements % 8;
    uint32_t i, j;
    uint32_t index = 0;

    __m256 v, scaleFactors, values;
    scaleFactors = _mm256_set1_ps(scale_factor);
    for (i = 0; i < moves; i++) {
        __m256i int32Values = _mm256_cvtepi8_epi32(*reinterpret_cast<const __m128i*>(&ptr_src[i * 8]));
        v = _mm256_cvtepi32_ps(int32Values);

        // values = v * 1/scaleFactors
        values = _mm256_div_ps(v, scaleFactors);
        if (transpose) {
            for (j = 0; j < 8; j++, index++) {
                uint32_t targetColumn = index / num_columns;
                uint32_t targetRow = index % num_columns;
                // target number of rows == source number of columns
                uint32_t targetIndex = targetRow * num_rows + targetColumn;
                ptr_dst[targetIndex] = values.m256_f32[j];
            }
        } else {
            _mm256_store_ps(&ptr_dst[i * 8], values);
        }
    }
    for (i = 0; i < mod; i++, index++) {
        if (transpose) {
            uint32_t targetColumn = index / num_columns;
            uint32_t targetRow = index % num_columns;
            // target number of rows == source number of columns
            uint32_t targetIndex = targetRow * num_rows + targetColumn;
            ptr_dst[targetIndex] = static_cast<float>(ptr_src[moves * 8 + i] / scale_factor);
        } else {
            ptr_dst[moves * 8 + i] = static_cast<float>(ptr_src[moves * 8 + i] / scale_factor);
        }
    }
}
#else

void GNAPluginNS::ConvertMatrixFp32ToInt16(int16_t* /*ptr_dst*/,
                                           const float* /*ptr_src*/,
                                           const uint32_t /*num_rows*/,
                                           const uint32_t /*num_columns*/,
                                           const float /*scale_factor*/,
                                           bool /*transpose*/) {}

void GNAPluginNS::ConvertMatrixFp32ToInt8(int8_t* /*ptr_dst*/,
                                          const float* /*ptr_src*/,
                                          const uint32_t /*num_rows*/,
                                          const uint32_t /*num_columns*/,
                                          const float /*scale_factor*/,
                                          bool /*transpose*/) {}

void GNAPluginNS::ConvertMatrixInt32ToFp32Avx(float* /*ptr_dst*/,
                                              const int32_t* /*ptr_src*/,
                                              uint32_t /*num_rows*/,
                                              uint32_t /*num_columns*/,
                                              float /*scale_factor*/,
                                              bool /*transpose*/) {}

void GNAPluginNS::ConvertMatrixInt16ToFp32Avx(float* /*ptr_dst*/,
                                              const int16_t* /*ptr_src*/,
                                              uint32_t /*num_rows*/,
                                              uint32_t /*num_columns*/,
                                              float /*scale_factor*/,
                                              bool /*transpose*/) {}

void GNAPluginNS::ConvertMatrixInt8ToFp32Avx(float* /*ptr_dst*/,
                                             const int8_t* /*ptr_src*/,
                                             uint32_t /*num_rows*/,
                                             uint32_t /*num_columns*/,
                                             float /*scale_factor*/,
                                             bool /*transpose*/) {}
#endif  // HAVE_AVX2
