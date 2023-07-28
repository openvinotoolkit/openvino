// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "gna_data_types.hpp"

namespace ov {
namespace intel_gna {
namespace pre_post_processing {

void ConvertToInt16(int16_t* ptr_dst,
                    const float* ptr_src,
                    const size_t num_rows,
                    const size_t num_columns,
                    const float scale_factor);

template <typename T>
T FloatToInt(float src) {
    float rounding_value = (src > 0) ? 0.5f : -0.5f;
    float value = src + rounding_value;
    if (value > static_cast<float>(std::numeric_limits<T>::max())) {
        return std::numeric_limits<T>::max();
    } else if (value < static_cast<float>(std::numeric_limits<T>::min())) {
        return std::numeric_limits<T>::min();
    }
    return static_cast<T>(value);
}

template <typename T, typename U>
inline void unscale_transpose_and_cast(T* ptr_dst,
                                       const U* ptr_src,
                                       intel_dnn_orientation_t orientation,
                                       size_t num_frames,
                                       size_t num_group,
                                       size_t num_vector_elements,
                                       size_t num_active_elements,
                                       size_t num_vector_stride,
                                       const float scale_factor) {
    // source scores are possibly padded to multiple of 8 and possibly interleaved
    // rotate if necessary and only copy actual scores (not padding)
    if (orientation == kDnnInterleavedOrientation) {
        for (size_t i = 0; i < num_frames; i++) {
            for (size_t j = 0; j < num_active_elements; j++) {
                ptr_dst[i * num_vector_elements + j] = static_cast<T>(ptr_src[j * num_group + i] / scale_factor);
            }
            for (size_t j = num_active_elements; j < num_vector_elements; j++) {
                ptr_dst[i * num_vector_elements + j] = 0;
            }
        }
    } else {
        for (size_t i = 0; i < num_frames; i++) {
            for (size_t j = 0; j < num_vector_elements; j++) {
                ptr_dst[i * num_vector_elements + j] =
                    static_cast<T>(ptr_src[i * num_vector_stride + j] / scale_factor);
            }
        }
    }
}

template <typename T, typename U>
void copy_input_data(T* dst,
                     const U* src,
                     size_t num_frames,
                     size_t num_group,
                     size_t num_vector_elements,
                     size_t num_vector_stride,
                     intel_dnn_orientation_t orientation,
                     float scale_factor) {
    if (!dst || !src) {
        return;
    }
    if (orientation == kDnnInterleavedOrientation) {
        for (size_t i = 0; i < num_frames; i++) {
            for (size_t j = 0; j < num_vector_elements; j++) {
                if (!std::is_same<T, U>::value) {
                    dst[j * num_group + i] = FloatToInt<T>(src[i * num_vector_elements + j] * scale_factor);
                } else {
                    dst[j * num_group + i] = static_cast<T>(src[i * num_vector_elements + j]);
                }
            }
            // pad to meet weight matrix row length requirement
            for (size_t j = num_vector_elements; j < num_vector_stride; j++) {
                dst[j * num_group + i] = 0;
            }
        }
        // pad partial group
        for (size_t i = num_frames; i < num_group; i++) {
            for (size_t j = 0; j < num_vector_stride; j++) {
                dst[j * num_group + i] = 0;
            }
        }
    } else {
        if (!std::is_same<T, U>::value) {
            for (size_t i = 0; i < num_frames; i++) {
                T* ptr_dst_vec = reinterpret_cast<T*>(dst) + i * num_vector_stride;
                const U* ptr_src_vec = reinterpret_cast<const U*>(src) + i * num_vector_elements;
                std::memset(ptr_dst_vec, 0, num_vector_stride * sizeof(T));
                for (size_t j = 0; j < num_vector_elements; j++) {
                    ptr_dst_vec[j] = FloatToInt<T>(ptr_src_vec[j] * scale_factor);
                }
            }
        } else {
            for (size_t i = 0; i < num_frames; i++) {
                void* ptr_dst_vec = reinterpret_cast<uint8_t*>(dst) + i * num_vector_stride * sizeof(T);
                const void* ptr_src_vec = reinterpret_cast<const uint8_t*>(src) + i * num_vector_elements * sizeof(U);
                std::memset(ptr_dst_vec, 0, num_vector_stride * sizeof(T));
                ie_memcpy(ptr_dst_vec, num_vector_elements * sizeof(T), ptr_src_vec, num_vector_elements * sizeof(T));
            }
        }

        for (size_t i = num_frames; i < num_group; i++) {
            void* ptr_dst_vec = reinterpret_cast<uint8_t*>(dst) + i * num_vector_stride * sizeof(T);
            std::memset(ptr_dst_vec, 0, num_vector_stride * sizeof(T));
        }
    }
}

}  // namespace pre_post_processing
}  // namespace intel_gna
}  // namespace ov