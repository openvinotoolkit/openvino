// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "dnnl_scratch_pad.h"
#include "openvino/core/type/float16.hpp"
#include "simd_jit.hpp"

namespace ov {
namespace intel_cpu {

class ActSparseFcKernel {
public:
    // compile time parameters
    ActSparseFcKernel(DnnlScratchPadPtr scrach_pad,
                      bool is_quantized,
                      bool is_int4,
                      bool with_zero_points,
                      int ic_group_size);

    void operator()(const float* input,
                    float* output,
                    int M,
                    int IC,
                    int OC,
                    float threshold,
                    float zero_point,
                    const void* W,
                    const float* scales,
                    const uint8_t* zp);

    void repack_weights_i4(uint8_t* src, uint8_t* dst, int IC, int OC);

private:
    void reduce_outputs(float* dst0, float* src0, int num_copies, int64_t OC);
    void
    MM_ComputeBounded_reuseA_f16(const float* A, float* C, const ov::float16* W, int M, int IC, int OC, int n0, int n1);
    void MM_ComputeBounded_reuseA_i8(const float* A,
                                     float* C,
                                     const uint8_t* W,
                                     const uint8_t* zp,
                                     const float* scales,
                                     int M,
                                     int IC,
                                     int OC,
                                     int64_t n0,
                                     int64_t n1);

    void MM_ComputeBounded_reuseB_i8(const float* A,
                                     float* C,
                                     const uint8_t* W,
                                     const uint8_t* zp,
                                     const float* scales,
                                     int M,
                                     int IC,
                                     int OC,
                                     int n0,
                                     int n1);

    void MM_ComputeBounded_reuseA_i4(const float* A,
                                     float* C,
                                     const uint8_t* W,
                                     const uint8_t* zp,
                                     const float* scales,
                                     int M,
                                     int IC,
                                     int OC,
                                     int n0,
                                     int n1,
                                     int icgs);
    void gemm6x2_Mx2(const float* pA,
                     int64_t A_stride,
                     const float* pB,
                     int64_t B_stride,
                     const float* pC,
                     int64_t C_stride,
                     int M,
                     int64_t bK,
                     int64_t is_accumulate_C);
    template <class T>
    T* scratch_alloc(size_t cnt);

    std::shared_ptr<SIMDJit> gemm6x2[6];
    std::shared_ptr<SIMDJit> gemm4x3[4];
    std::shared_ptr<SIMDJit> gemm4x1[4];
    std::shared_ptr<SIMDJit> m_decompzp_kernel;
    std::shared_ptr<SIMDJit> m_accumulate_kernel;
    std::shared_ptr<SIMDJit> m_reduce_outputs_kernel;
    std::shared_ptr<SIMDJit> m_repack_2xsimdw_kernel;
    std::shared_ptr<SIMDJit> m_repack_3xsimdw_i8_kernel;

    const bool m_is_quantized;
    const bool m_is_int4;
    const bool m_with_zp;
    const int m_ic_group_size;

    std::vector<int> m_nonzero_ids;
    std::vector<float> m_nonzero_val;
    std::vector<float> m_output_temp;
    int m_nonzero_cnt;

    DnnlScratchPadPtr m_scrach_pad;
    MemoryPtr m_scratch_mem;
    uint8_t* m_scratch_base = nullptr;
};

}  // namespace intel_cpu
}  // namespace ov
