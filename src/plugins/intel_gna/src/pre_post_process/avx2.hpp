// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_system_conf.h>
#include <stdint.h>

namespace ov {
namespace intel_gna {

inline bool isAvx2Supported() {
#ifdef HAVE_AVX2
    return InferenceEngine::with_cpu_x86_avx2();
#else
    return false;
#endif  // HAVE_AVX2
}

namespace pre_post_processing {

void convert_matrix_fp32_to_int16_avx(int16_t* ptr_dst,
                                      const float* ptr_src,
                                      const size_t num_rows,
                                      const size_t num_columns,
                                      const float scale_factor,
                                      bool transpose);

void convert_matrix_fp32_to_int8_avx(int8_t* ptr_dst,
                                     const float* ptr_src,
                                     const size_t num_rows,
                                     const size_t num_columns,
                                     const float scale_factor,
                                     bool transpose);

void convert_matrix_int32_to_fp32_avx(float* ptr_dst,
                                      const int32_t* ptr_src,
                                      size_t num_rows,
                                      size_t num_columns,
                                      float scale_factor,
                                      bool transpose);

void convert_matrix_int16_to_fp32_avx(float* ptr_dst,
                                      const int16_t* ptr_src,
                                      size_t num_rows,
                                      size_t num_columns,
                                      float scale_factor,
                                      bool transpose);

void convert_matrix_int8_to_fp32_avx(float* ptr_dst,
                                     const int8_t* ptr_src,
                                     size_t num_rows,
                                     size_t num_columns,
                                     float scale_factor,
                                     bool transpose);
}  // namespace pre_post_processing
}  // namespace intel_gna
}  // namespace ov