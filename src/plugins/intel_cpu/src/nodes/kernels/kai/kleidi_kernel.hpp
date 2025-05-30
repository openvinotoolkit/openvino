// Copyright (C) 2025 FUJITSU LIMITED
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <arm_neon.h>
#include <kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.h>
#include <kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p_interface.h>
#include <kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.h>

#include <limits>
#include <openvino/core/type/element_type.hpp>

namespace ov::intel_cpu {

class KleidiGemm {
public:
    KleidiGemm(size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc);
    void executeGemm(const void* a, const void* b, void* c);
    void packB(const float16_t* inp, const float16_t* bias, float16_t* packed_out);
    const size_t get_packed_rhs_size() const;

private:
    static constexpr kai_matmul_clamp_f16_f16_f16p_ukernel ukernel{
        kai_get_m_step_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_get_n_step_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_get_nr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_get_kr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_get_sr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_get_lhs_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_get_rhs_packed_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_get_dst_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_get_dst_size_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_run_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla};
    size_t M, N, K;
    size_t lda, ldb, ldc;
    size_t nr, kr, sr;
    size_t packedRHSsize;
};

KleidiGemm::KleidiGemm(size_t _M, size_t _N, size_t _K, size_t _lda, size_t _ldb, size_t _ldc)
    : M(_M),
      N(_N),
      K(_K),
      lda(_lda),
      ldb(_ldb),
      ldc(_ldc),
      nr(ukernel.get_nr()),
      kr(ukernel.get_kr()),
      sr(ukernel.get_sr()),
      packedRHSsize(kai_get_rhs_packed_size_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(_N, _K)){};

const size_t KleidiGemm::get_packed_rhs_size() const {
    return packedRHSsize;
}

void KleidiGemm::packB(const float16_t* inp, const float16_t* bias, float16_t* packed_out) {
    // Packing only needs to be performed once if the contents of the bias and RHS matrices are expected to be constant.
    kai_run_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(1,
                                                      N,
                                                      K,
                                                      nr,
                                                      kr,
                                                      sr,                       // Packing arguments
                                                      ldb * sizeof(float16_t),  // RHS stride
                                                      inp,                      // RHS
                                                      bias,                     // Bias
                                                      NULL,                     // Scale
                                                      packed_out,               // RHS packed
                                                      0,
                                                      NULL);
}

void KleidiGemm::executeGemm(const void* a, const void* b, void* c) {
    const size_t m_step = ukernel.get_m_step();
    const size_t n_step = ukernel.get_n_step();
    for (size_t i_m_step = 0; i_m_step < M; i_m_step += m_step) {
        for (size_t i_n_step = 0; i_n_step < N; i_n_step += n_step) {
            const uint8_t* lhs_ptr =
                static_cast<const uint8_t*>(a) + (ukernel.get_lhs_packed_offset(i_m_step, lda * sizeof(float16_t)));
            const uint8_t* rhs_ptr = static_cast<const uint8_t*>(b) + (ukernel.get_rhs_packed_offset(i_n_step, K));
            uint8_t* dst_ptr =
                static_cast<uint8_t*>(c) + (ukernel.get_dst_offset(i_m_step, i_n_step, ldc * sizeof(float16_t)));
            const size_t actual_m = std::min(M - i_m_step, m_step);
            const size_t actual_n = std::min(N - i_n_step, n_step);

            ukernel.run_matmul(actual_m,
                               actual_n,
                               K,                        // Dimensions
                               lhs_ptr,                  // LHS
                               lda * sizeof(float16_t),  // LHS stride
                               rhs_ptr,                  // RHS packed
                               dst_ptr,                  // DST
                               ldc * sizeof(float16_t),  // DST stride (row)
                               sizeof(float16_t),        // DST stride (col)
                               -std::numeric_limits<float>::max(),
                               std::numeric_limits<float>::max()  // Min and max for the clamp operation
            );
        }
    }
}

}  // namespace ov::intel_cpu