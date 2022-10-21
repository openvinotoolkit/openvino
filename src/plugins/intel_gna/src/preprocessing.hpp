// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "gna_data_types.hpp"
#include "common/avx_support.hpp"

namespace GNAPluginNS {

void ConvertToInt16(int16_t* ptr_dst,
                    const float* ptr_src,
                    const uint32_t num_rows,
                    const uint32_t num_columns,
                    const float scale_factor);

#ifdef HAVE_AVX2
void ConvertMatrixFp32ToInt16(int16_t* ptr_dst,
                               const float* ptr_src,
                               const uint32_t num_rows,
                               const uint32_t num_columns,
                               const float scale_factor,
                               bool transpose);

void ConvertMatrixFp32ToInt8(int8_t* ptr_dst,
                              const float* ptr_src,
                              const uint32_t num_rows,
                              const uint32_t num_columns,
                              const float scale_factor,
                              bool transpose);

void ConvertMatrixInt32ToFp32Avx(float* ptr_dst,
                                 const int32_t* ptr_src,
                                 uint32_t num_rows,
                                 uint32_t num_columns,
                                 float scale_factor,
                                 bool transpose);

void ConvertMatrixInt16ToFp32Avx(float* ptr_dst,
                                 const int16_t* ptr_src,
                                 uint32_t num_rows,
                                 uint32_t num_columns,
                                 float scale_factor,
                                 bool transpose);

void ConvertMatrixInt8ToFp32Avx(float* ptr_dst,
                                const int8_t* ptr_src,
                                uint32_t num_rows,
                                uint32_t num_columns,
                                float scale_factor,
                                bool transpose);
#endif // HAVE_AVX2

int16_t ConvertFloatToInt16(float src);
int8_t ConvertFloatToInt8(float src);

template <typename T, typename U>
inline void UnscaleTransposeAndCast(T* ptr_dst,
                                    const U* ptr_src,
                                    intel_dnn_orientation_t orientation,
                                    uint32_t num_frames,
                                    uint32_t num_group,
                                    uint32_t num_vector_elements,
                                    uint32_t num_active_elements,
                                    uint32_t num_vector_stride,
                                    const float scale_factor) {
    // source scores are possibly padded to multiple of 8 and possibly interleaved
    // rotate if necessary and only copy actual scores (not padding)
    if (orientation == kDnnInterleavedOrientation) {
        for (uint32_t i = 0; i < num_frames; i++) {
            for (uint32_t j = 0; j < num_active_elements; j++) {
                ptr_dst[i * num_vector_elements + j] = static_cast<T>(ptr_src[j * num_group + i] / scale_factor);
            }
            for (uint32_t j = num_active_elements; j < num_vector_elements; j++) {
                ptr_dst[i * num_vector_elements + j] = 0;
            }
        }
    } else {
        for (uint32_t i = 0; i < num_frames; i++) {
            for (uint32_t j = 0; j < num_vector_elements; j++) {
                ptr_dst[i * num_vector_elements + j] =
                    static_cast<T>(ptr_src[i * num_vector_stride + j] / scale_factor);
            }
        }
    }
}

}  // namespace GNAPluginNS
