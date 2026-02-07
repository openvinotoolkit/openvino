// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/kernels/aarch64/brgemm_int8_kernel.hpp"

#include <common/c_types_map.hpp>
#include <cpu/aarch64/brgemm/brgemm.hpp>
#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <openvino/core/except.hpp>

#include <algorithm>

#if defined(__linux__)
#    include <sys/auxv.h>
#endif
#if defined(__linux__) && defined(__aarch64__)
#    include <asm/hwcap.h>
#endif

#include "nodes/kernels/aarch64/jit_int8_conv_kernel.hpp"

namespace ov::intel_cpu::aarch64 {

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::aarch64;

namespace {
bool has_asimd_dotprod() {
#if defined(__linux__) && defined(__aarch64__) && defined(HWCAP_ASIMDDP)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#else
    return false;
#endif
}
}  // namespace

void BrgemmInt8Kernel::BrgKernelDeleter::operator()(brgemm_kernel_t* ptr) const noexcept {
    if (ptr) {
        brgemm_kernel_destroy(ptr);
    }
}

BrgemmInt8Kernel::BrgemmInt8Kernel(size_t M,
                                   size_t N,
                                   size_t K,
                                   size_t lda,
                                   size_t ldb,
                                   size_t ldc,
                                   bool src_signed) {
    M_ = M;
    N_ = N;
    K_ = K;
    lda_ = lda;
    ldb_ = ldb;
    ldb_fallback_ = K;
    ldc_ = ldc;
    src_signed_ = src_signed;

    const auto isa = []() {
        if (mayiuse(sve_512)) {
            return cpu_isa_t::sve_512;
        }
        if (mayiuse(sve_256)) {
            return cpu_isa_t::sve_256;
        }
        return cpu_isa_t::sve_128;
    }();

    const auto dt_a = src_signed ? data_type::s8 : data_type::u8;
    const auto dt_b = data_type::s8;

    brgemm_t brg = {};
    auto st = brgemm_desc_init(&brg,
                                   isa,
                                   brgemm_addr,
                                   dt_a,
                                   dt_b,
                                   false,
                                   false,
                                   brgemm_row_major,
                                   1.0f,
                                   0.0f,
                                   static_cast<dim_t>(lda),
                                   static_cast<dim_t>(ldb),
                                   static_cast<dim_t>(ldc),
                                   static_cast<dim_t>(M),
                                   static_cast<dim_t>(N),
                                   static_cast<dim_t>(K),
                                   nullptr);
    if (st == status::success) {
        st = brgemm_desc_finalize(&brg);
    }
    if (st == status::success) {
        brgemm_kernel_t* kernel = nullptr;
        st = brgemm_kernel_create(&kernel, brg);
        if (st == status::success && kernel) {
            kernel_.reset(kernel);
            use_brgemm_ = true;
            return;
        }
    }

    has_dotprod_ = has_asimd_dotprod();
    kernel_1x4_ = std::make_unique<jit_int8_brgemm_kernel_1x4>(src_signed_);
    kernel_1x4_->create_ker();
    if (has_dotprod_) {
        if (src_signed_) {
            kernel_1x4_dot_ = std::make_unique<jit_int8_brgemm_kernel_1x4_dot>();
            kernel_1x4_dot_->create_ker();
            kernel_1x8_dot_ = std::make_unique<jit_int8_brgemm_kernel_1x8_dot>();
            kernel_1x8_dot_->create_ker();
            kernel_2x8_dot_ = std::make_unique<jit_int8_brgemm_kernel_2x8_dot>();
            kernel_2x8_dot_->create_ker();
            kernel_4x4_dot_ = std::make_unique<jit_int8_brgemm_kernel_4x4_dot>();
            kernel_4x4_dot_->create_ker();
        } else {
            kernel_1x4_udot_ = std::make_unique<jit_int8_brgemm_kernel_1x4_udot>();
            kernel_1x4_udot_->create_ker();
            kernel_1x8_udot_ = std::make_unique<jit_int8_brgemm_kernel_1x8_udot>();
            kernel_1x8_udot_->create_ker();
            kernel_2x8_udot_ = std::make_unique<jit_int8_brgemm_kernel_2x8_udot>();
            kernel_2x8_udot_->create_ker();
            kernel_4x4_udot_ = std::make_unique<jit_int8_brgemm_kernel_4x4_udot>();
            kernel_4x4_udot_->create_ker();
        }
    }
}

BrgemmInt8Kernel::~BrgemmInt8Kernel() = default;

void BrgemmInt8Kernel::execute(const void* a, const int8_t* b, int32_t* c) const {
    brgemm_batch_element_t addr_batch;
    addr_batch.ptr.A = a;
    addr_batch.ptr.B = b;
    if (use_brgemm_) {
        brgemm_kernel_execute(kernel_.get(), 1, &addr_batch, c, nullptr);
        return;
    }
    execute_fallback(&addr_batch, 1, c);
}

void BrgemmInt8Kernel::execute_batch(const brgemm_batch_element_t* batch, int bs, int32_t* c) const {
    if (use_brgemm_) {
        brgemm_kernel_execute(kernel_.get(), bs, batch, c, nullptr);
        return;
    }
    execute_fallback(batch, bs, c);
}

void BrgemmInt8Kernel::execute_fallback(const brgemm_batch_element_t* batch, int bs, int32_t* c) const {
    if (!batch || bs <= 0 || M_ == 0 || N_ == 0 || K_ == 0) {
        return;
    }

    const size_t M = M_;
    const size_t N = N_;
    const size_t K = K_;
    const size_t lda = lda_;
    const size_t ldb = ldb_fallback_;
    const size_t ldc = ldc_;
    const bool can_use_dot = src_signed_ && has_dotprod_ && kernel_1x4_dot_ && kernel_4x4_dot_;
    const bool can_use_dot8 = src_signed_ && has_dotprod_ && kernel_1x8_dot_ && kernel_2x8_dot_;
    const bool can_use_udot = !src_signed_ && has_dotprod_ && kernel_1x4_udot_;
    const bool can_use_udot8 = !src_signed_ && has_dotprod_ && kernel_1x8_udot_ && kernel_2x8_udot_;
    const bool can_use_udot4x4 = !src_signed_ && has_dotprod_ && kernel_4x4_udot_;

    for (size_t m0 = 0; m0 < M; m0 += 4) {
        const size_t m_block = std::min<size_t>(4, M - m0);
        size_t n0 = 0;
        while (n0 < N) {
            const size_t n_step = (can_use_dot8 && (N - n0) >= 8) ? 8 : 4;
            const size_t n_block = std::min<size_t>(n_step, N - n0);

            if (m_block == 4 && n_block == 4) {
                if (can_use_dot) {
                    for (int b = 0; b < bs; ++b) {
                        const auto* src = static_cast<const int8_t*>(batch[b].ptr.A);
                        const auto* wei = static_cast<const int8_t*>(batch[b].ptr.B);
                        const int8_t* srcs[4] = {
                            src + (m0 + 0) * lda,
                            src + (m0 + 1) * lda,
                            src + (m0 + 2) * lda,
                            src + (m0 + 3) * lda,
                        };
                        int32_t* dst = c + m0 * ldc + n0;
                        kernel_4x4_dot_->ker()(srcs,
                                               wei + n0 * ldb,
                                               dst,
                                               K,
                                               ldb,
                                               ldc * sizeof(int32_t),
                                               b == 0 ? 0 : 1);
                    }
                    n0 += n_block;
                    continue;
                }
                if (can_use_udot4x4) {
                    for (int b = 0; b < bs; ++b) {
                        const auto* src = static_cast<const uint8_t*>(batch[b].ptr.A);
                        const auto* wei = static_cast<const int8_t*>(batch[b].ptr.B);
                        const uint8_t* srcs[4] = {
                            src + (m0 + 0) * lda,
                            src + (m0 + 1) * lda,
                            src + (m0 + 2) * lda,
                            src + (m0 + 3) * lda,
                        };
                        int32_t* dst = c + m0 * ldc + n0;
                        kernel_4x4_udot_->ker()(srcs,
                                                wei + n0 * ldb,
                                                dst,
                                                K,
                                                ldb,
                                                ldc * sizeof(int32_t),
                                                b == 0 ? 0 : 1);
                    }
                    n0 += n_block;
                    continue;
                }
            }

            if (n_block == 8 && can_use_dot8) {
                for (int b = 0; b < bs; ++b) {
                    const int accum = b == 0 ? 0 : 1;
                    const auto* src = static_cast<const int8_t*>(batch[b].ptr.A);
                    const auto* wei = static_cast<const int8_t*>(batch[b].ptr.B);
                    size_t r = 0;
                    for (; r + 1 < m_block; r += 2) {
                        const int8_t* srcs[2] = {
                            src + (m0 + r) * lda,
                            src + (m0 + r + 1) * lda,
                        };
                        int32_t* dst_row = c + (m0 + r) * ldc + n0;
                        kernel_2x8_dot_->ker()(srcs,
                                               wei + n0 * ldb,
                                               dst_row,
                                               K,
                                               ldb,
                                               ldc * sizeof(int32_t),
                                               accum);
                    }
                    if (r < m_block) {
                        const int8_t* src_row = src + (m0 + r) * lda;
                        int32_t* dst_row = c + (m0 + r) * ldc + n0;
                        kernel_1x8_dot_->ker()(src_row, wei + n0 * ldb, dst_row, K, ldb, accum);
                    }
                }
            } else if (n_block == 8 && can_use_udot8) {
                for (int b = 0; b < bs; ++b) {
                    const int accum = b == 0 ? 0 : 1;
                    const auto* src = static_cast<const uint8_t*>(batch[b].ptr.A);
                    const auto* wei = static_cast<const int8_t*>(batch[b].ptr.B);
                    size_t r = 0;
                    for (; r + 1 < m_block; r += 2) {
                        const uint8_t* srcs[2] = {
                            src + (m0 + r) * lda,
                            src + (m0 + r + 1) * lda,
                        };
                        int32_t* dst_row = c + (m0 + r) * ldc + n0;
                        kernel_2x8_udot_->ker()(srcs,
                                                wei + n0 * ldb,
                                                dst_row,
                                                K,
                                                ldb,
                                                ldc * sizeof(int32_t),
                                                accum);
                    }
                    if (r < m_block) {
                        const uint8_t* src_row = src + (m0 + r) * lda;
                        int32_t* dst_row = c + (m0 + r) * ldc + n0;
                        kernel_1x8_udot_->ker()(src_row, wei + n0 * ldb, dst_row, K, ldb, accum);
                    }
                }
            } else {
                int32_t tmp[16];
                std::fill(tmp, tmp + 16, 0);
                for (int b = 0; b < bs; ++b) {
                    const int accum = b == 0 ? 0 : 1;
                    const auto* wei = static_cast<const int8_t*>(batch[b].ptr.B);
                    if (src_signed_) {
                        const auto* src = static_cast<const int8_t*>(batch[b].ptr.A);
                        for (size_t r = 0; r < m_block; ++r) {
                            const int8_t* src_row = src + (m0 + r) * lda;
                            int32_t* dst_row = tmp + r * 4;
                            if (has_dotprod_ && kernel_1x4_dot_) {
                                kernel_1x4_dot_->ker()(src_row, wei + n0 * ldb, dst_row, K, ldb, accum);
                            } else {
                                kernel_1x4_->ker()(reinterpret_cast<const uint8_t*>(src_row),
                                                   wei + n0 * ldb,
                                                   dst_row,
                                                   K,
                                                   ldb,
                                                   accum);
                            }
                        }
                    } else {
                        const auto* src = static_cast<const uint8_t*>(batch[b].ptr.A);
                        for (size_t r = 0; r < m_block; ++r) {
                            const uint8_t* src_row = src + (m0 + r) * lda;
                            int32_t* dst_row = tmp + r * 4;
                            if (can_use_udot) {
                                kernel_1x4_udot_->ker()(src_row, wei + n0 * ldb, dst_row, K, ldb, accum);
                            } else {
                                kernel_1x4_->ker()(src_row, wei + n0 * ldb, dst_row, K, ldb, accum);
                            }
                        }
                    }
                }

                for (size_t r = 0; r < m_block; ++r) {
                    int32_t* dst_row = c + (m0 + r) * ldc + n0;
                    const int32_t* src_row = tmp + r * 4;
                    for (size_t n = 0; n < n_block; ++n) {
                        dst_row[n] = src_row[n];
                    }
                }
            }
            n0 += n_block;
        }
    }
}

}  // namespace ov::intel_cpu::aarch64
