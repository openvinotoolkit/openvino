// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

namespace ov {
namespace intel_gna {
namespace pre_post_processing {

class DataStorageConverter {
public:
    virtual void convert_matrix_fp32_to_int16_avx(int16_t* ptr_dst,
                                                  const float* ptr_src,
                                                  const size_t num_rows,
                                                  const size_t num_columns,
                                                  const float scale_factor,
                                                  bool transpose) = 0;
    virtual void convert_matrix_fp32_to_int8_avx(int8_t* ptr_dst,
                                                 const float* ptr_src,
                                                 const size_t num_rows,
                                                 const size_t num_columns,
                                                 const float scale_factor,
                                                 bool transpose) = 0;
    virtual void convert_matrix_int32_to_fp32_avx(float* ptr_dst,
                                                  const int32_t* ptr_src,
                                                  size_t num_rows,
                                                  size_t num_columns,
                                                  float scale_factor,
                                                  bool transpose) = 0;
    virtual void convert_matrix_int16_to_fp32_avx(float* ptr_dst,
                                                  const int16_t* ptr_src,
                                                  size_t num_rows,
                                                  size_t num_columns,
                                                  float scale_factor,
                                                  bool transpose) = 0;
    virtual void convert_matrix_int8_to_fp32_avx(float* ptr_dst,
                                                 const int8_t* ptr_src,
                                                 size_t num_rows,
                                                 size_t num_columns,
                                                 float scale_factor,
                                                 bool transpose) = 0;
};

#ifdef HAVE_AVX2

class DataStorageConverterAvx : public DataStorageConverter {
public:
    void convert_matrix_fp32_to_int16_avx(int16_t* ptr_dst,
                                          const float* ptr_src,
                                          const size_t num_rows,
                                          const size_t num_columns,
                                          const float scale_factor,
                                          bool transpose) override;

    void convert_matrix_fp32_to_int8_avx(int8_t* ptr_dst,
                                         const float* ptr_src,
                                         const size_t num_rows,
                                         const size_t num_columns,
                                         const float scale_factor,
                                         bool transpose) override;

    void convert_matrix_int32_to_fp32_avx(float* ptr_dst,
                                          const int32_t* ptr_src,
                                          size_t num_rows,
                                          size_t num_columns,
                                          float scale_factor,
                                          bool transpose) override;

    void convert_matrix_int16_to_fp32_avx(float* ptr_dst,
                                          const int16_t* ptr_src,
                                          size_t num_rows,
                                          size_t num_columns,
                                          float scale_factor,
                                          bool transpose) override;

    void convert_matrix_int8_to_fp32_avx(float* ptr_dst,
                                         const int8_t* ptr_src,
                                         size_t num_rows,
                                         size_t num_columns,
                                         float scale_factor,
                                         bool transpose) override;
};

#endif  // HAVE_AVX2

}  // namespace pre_post_processing
}  // namespace intel_gna
}  // namespace ov