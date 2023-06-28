// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "avx2.hpp"

#ifdef HAVE_AVX2
#    include <immintrin.h>

#    include "preprocessing.hpp"
#endif  // HAVE_AVX2

namespace ov {
namespace intel_gna {
namespace pre_post_processing {

#ifdef HAVE_AVX2

template <typename T>
void convert_matrix_fp32_to_int_avx(T* ptr_dst,
                                    const float* ptr_src,
                                    const size_t num_rows,
                                    const size_t num_columns,
                                    const float scale_factor,
                                    bool transpose) {
    const size_t num_elements = num_rows * num_columns;
    size_t moves = num_elements / 8;
    size_t mod = num_elements % 8;
    size_t i, j;
    size_t index = 0;

    __m256 v, zero, half, neg_half, scaleFactors, mask, roundingValues, min, max, values;

    zero = _mm256_setzero_ps();
    half = _mm256_set1_ps(0.5f);
    neg_half = _mm256_set1_ps(-0.5f);
    scaleFactors = _mm256_set1_ps(scale_factor);
    max = _mm256_set1_ps(static_cast<float>(std::numeric_limits<T>::max()));
    min = _mm256_set1_ps(static_cast<float>(std::numeric_limits<T>::min()));

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
                size_t targetColumn = index / num_columns;
                size_t targetRow = index % num_columns;
                // target number of rows == source number of columns
                size_t targetIndex = targetRow * num_rows + targetColumn;
                ptr_dst[targetIndex] = static_cast<T>(values.m256_f32[j]);
            }
        } else {
            for (j = 0; j < 8; j++)
                ptr_dst[i * 8 + j] = static_cast<T>(values.m256_f32[j]);
        }
    }

    for (i = 0; i < mod; i++, index++) {
        if (transpose) {
            size_t targetColumn = index / num_columns;
            size_t targetRow = index % num_columns;
            // target number of rows == source number of columns
            size_t targetIndex = targetRow * num_rows + targetColumn;
            ptr_dst[targetIndex] = FloatToInt<T>(ptr_src[moves * 8 + i] * scale_factor);
        } else {
            ptr_dst[moves * 8 + i] = FloatToInt<T>(ptr_src[moves * 8 + i] * scale_factor);
        }
    }
}

void convert_matrix_fp32_to_int16_avx(int16_t* ptr_dst,
                                      const float* ptr_src,
                                      const size_t num_rows,
                                      const size_t num_columns,
                                      const float scale_factor,
                                      bool transpose) {
    convert_matrix_fp32_to_int_avx(ptr_dst, ptr_src, num_rows, num_columns, scale_factor, transpose);
}

void convert_matrix_fp32_to_int8_avx(int8_t* ptr_dst,
                                     const float* ptr_src,
                                     const size_t num_rows,
                                     const size_t num_columns,
                                     const float scale_factor,
                                     bool transpose) {
    convert_matrix_fp32_to_int_avx(ptr_dst, ptr_src, num_rows, num_columns, scale_factor, transpose);
}


template <typename T>
void convert_matrix_int_to_fp32_avx(float* ptr_dst,
                                     const T* ptr_src,
                                     size_t num_rows,
                                     size_t num_columns,
                                     float scale_factor,
                                     bool transpose) {
    const size_t num_elements = num_rows * num_columns;
    size_t moves = num_elements / 8;
    size_t mod = num_elements % 8;
    size_t i, j;
    size_t index = 0;

    __m256 v, scaleFactors, values;
    scaleFactors = _mm256_set1_ps(scale_factor);
    for (i = 0; i < moves; i++) {
        if (std::is_same<T, int8_t>::value) {
            __m256i int32Values = _mm256_cvtepi8_epi32(*reinterpret_cast<const __m128i*>(&ptr_src[i * 8]));
            v = _mm256_cvtepi32_ps(int32Values);
        } else if (std::is_same<T, int16_t>::value) {
            __m256i int32Values = _mm256_cvtepi16_epi32(*reinterpret_cast<const __m128i*>(&ptr_src[i * 8]));
            v = _mm256_cvtepi32_ps(int32Values);
        } else if (std::is_same<T, int32_t>::value) {
            v = _mm256_cvtepi32_ps(*reinterpret_cast<const __m256i*>(&ptr_src[i * 8]));
        }

        // values = v * 1/scaleFactors
        values = _mm256_div_ps(v, scaleFactors);
        if (transpose) {
            for (j = 0; j < 8; j++, index++) {
                size_t targetColumn = index / num_columns;
                size_t targetRow = index % num_columns;
                // target number of rows == source number of columns
                size_t targetIndex = targetRow * num_rows + targetColumn;
                ptr_dst[targetIndex] = values.m256_f32[j];
            }
        } else {
            _mm256_store_ps(&ptr_dst[i * 8], values);
        }
    }
    for (i = 0; i < mod; i++, index++) {
        if (transpose) {
            size_t targetColumn = index / num_columns;
            size_t targetRow = index % num_columns;
            // target number of rows == source number of columns
            size_t targetIndex = targetRow * num_rows + targetColumn;
            ptr_dst[targetIndex] = static_cast<float>(ptr_src[moves * 8 + i] / scale_factor);
        } else {
            ptr_dst[moves * 8 + i] = static_cast<float>(ptr_src[moves * 8 + i] / scale_factor);
        }
    }
}

void convert_matrix_int32_to_fp32_avx(float* ptr_dst,
    const int32_t* ptr_src,
    size_t num_rows,
    size_t num_columns,
    float scale_factor,
    bool transpose) {
    convert_matrix_int_to_fp32_avx(ptr_dst, ptr_src, num_rows, num_columns, scale_factor, transpose);
}

void convert_matrix_int16_to_fp32_avx(float* ptr_dst,
    const int16_t* ptr_src,
    size_t num_rows,
    size_t num_columns,
    float scale_factor,
    bool transpose) {
    convert_matrix_int_to_fp32_avx(ptr_dst, ptr_src, num_rows, num_columns, scale_factor, transpose);
}

void convert_matrix_int8_to_fp32_avx(float* ptr_dst,
                                     const int8_t* ptr_src,
                                     size_t num_rows,
                                     size_t num_columns,
                                     float scale_factor,
                                     bool transpose) {
    convert_matrix_int_to_fp32_avx(ptr_dst, ptr_src, num_rows, num_columns, scale_factor, transpose);
}

#else

void convert_matrix_fp32_to_int16_avx(int16_t* /*ptr_dst*/,
                                      const float* /*ptr_src*/,
                                      const size_t /*num_rows*/,
                                      const size_t /*num_columns*/,
                                      const float /*scale_factor*/,
                                      bool /*transpose*/) {}

void convert_matrix_fp32_to_int8_avx(int8_t* /*ptr_dst*/,
                                     const float* /*ptr_src*/,
                                     const size_t /*num_rows*/,
                                     const size_t /*num_columns*/,
                                     const float /*scale_factor*/,
                                     bool /*transpose*/) {}

void convert_matrix_int32_to_fp32_avx(float* /*ptr_dst*/,
                                      const int32_t* /*ptr_src*/,
                                      size_t /*num_rows*/,
                                      size_t /*num_columns*/,
                                      float /*scale_factor*/,
                                      bool /*transpose*/) {}

void convert_matrix_int16_to_fp32_avx(float* /*ptr_dst*/,
                                      const int16_t* /*ptr_src*/,
                                      size_t /*num_rows*/,
                                      size_t /*num_columns*/,
                                      float /*scale_factor*/,
                                      bool /*transpose*/) {}

void convert_matrix_int8_to_fp32_avx(float* /*ptr_dst*/,
                                     const int8_t* /*ptr_src*/,
                                     size_t /*num_rows*/,
                                     size_t /*num_columns*/,
                                     float /*scale_factor*/,
                                     bool /*transpose*/) {}
#endif  // HAVE_AVX2

}  // namespace pre_post_processing
}  // namespace intel_gna
}  // namespace ov