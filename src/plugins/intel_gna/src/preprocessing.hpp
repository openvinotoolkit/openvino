// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "avx2.hpp"
#include "gna_data_types.hpp"

namespace GNAPluginNS {

void ConvertToInt16(int16_t* ptr_dst,
                    const float* ptr_src,
                    const uint32_t num_rows,
                    const uint32_t num_columns,
                    const float scale_factor);

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

template <typename T, typename U>
void CopyInputData(T* dst,
                   const U* src,
                   uint32_t num_frames,
                   uint32_t num_group,
                   uint32_t num_vector_elements,
                   uint32_t num_vector_stride,
                   intel_dnn_orientation_t orientation,
                   float scaleFactor,
                   bool input_low_precision) {
    if (!dst || !src) {
        return;
    }
    if (orientation == kDnnInterleavedOrientation) {
        for (uint32_t i = 0; i < num_frames; i++) {
            for (uint32_t j = 0; j < num_vector_elements; j++) {
                if (!std::is_same<T, U>::value) {
                    if (!input_low_precision) {
                        dst[j * num_group + i] = ConvertFloatToInt16(src[i * num_vector_elements + j] * scaleFactor);
                    } else {
                        dst[j * num_group + i] = ConvertFloatToInt8(src[i * num_vector_elements + j] * scaleFactor);
                    }
                } else {
                    dst[j * num_group + i] = src[i * num_vector_elements + j];
                }
            }
            // pad to meet weight matrix row length requirement
            for (uint32_t j = num_vector_elements; j < num_vector_stride; j++) {
                dst[j * num_group + i] = 0;
            }
        }
        // pad partial group
        for (uint32_t i = num_frames; i < num_group; i++) {
            for (uint32_t j = 0; j < num_vector_stride; j++) {
                dst[j * num_group + i] = 0;
            }
        }
    } else {
        if (!std::is_same<T, U>::value) {
            for (uint32_t i = 0; i < num_frames; i++) {
                T* ptr_dst_vec = reinterpret_cast<T*>(dst) + i * num_vector_stride;
                const U* ptr_src_vec = reinterpret_cast<const U*>(src) + i * num_vector_elements;
                std::memset(ptr_dst_vec, 0, num_vector_stride * sizeof(T));
                if (!input_low_precision) {
                    for (uint32_t j = 0; j < num_vector_elements; j++) {
                        ptr_dst_vec[j] = ConvertFloatToInt16(ptr_src_vec[j] * scaleFactor);
                    }
                } else {
                    for (uint32_t j = 0; j < num_vector_elements; j++) {
                        ptr_dst_vec[j] = ConvertFloatToInt8(ptr_src_vec[j] * scaleFactor);
                    }
                }
            }
        } else {
            for (uint32_t i = 0; i < num_frames; i++) {
                void* ptr_dst_vec = reinterpret_cast<uint8_t*>(dst) + i * num_vector_stride * sizeof(T);
                const void* ptr_src_vec = reinterpret_cast<const uint8_t*>(src) + i * num_vector_elements * sizeof(U);
                std::memset(ptr_dst_vec, 0, num_vector_stride * sizeof(T));
                ie_memcpy(ptr_dst_vec, num_vector_elements * sizeof(T), ptr_src_vec, num_vector_elements * sizeof(T));
            }
        }

        for (uint32_t i = num_frames; i < num_group; i++) {
            void* ptr_dst_vec = reinterpret_cast<uint8_t*>(dst) + i * num_vector_stride * sizeof(T);
            std::memset(ptr_dst_vec, 0, num_vector_stride * sizeof(T));
        }
    }
}

void ImportFrames(void* ptr_dst,
                  const void* ptr_src,
                  const InferenceEngine::Precision& input_precision,
                  float scaleFactor,
                  intel_dnn_orientation_t orientation,
                  uint32_t num_frames,
                  uint32_t num_group,
                  uint32_t num_vector_elements,
                  uint32_t num_vector_stride,
                  bool input_low_precision,
                  bool isGnaDevice,
                  bool isAvx2Supported);

void ExportScores(void* ptr_dst,
                  const void* ptr_src,
                  intel_dnn_orientation_t orientation,
                  uint32_t num_frames,
                  uint32_t num_group,
                  uint32_t num_vector_elements,
                  uint32_t num_active_elements,
                  uint32_t num_vector_stride,
                  const InferenceEngine::Precision& precision_in,
                  const InferenceEngine::Precision& precision_out,
                  const float scale_factor,
                  bool isAvx2Supported);
}  // namespace GNAPluginNS
