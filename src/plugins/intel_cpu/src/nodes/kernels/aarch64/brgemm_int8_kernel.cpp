// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/kernels/aarch64/brgemm_int8_kernel.hpp"

#include <common/c_types_map.hpp>
#include <cpu/aarch64/brgemm/brgemm.hpp>
#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <openvino/core/except.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include "nodes/kernels/aarch64/brgemm_kernels/int8_brgemm_kernels.hpp"

namespace ov::intel_cpu::aarch64 {

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::aarch64;

namespace {
cpu_isa_t get_preferred_brgemm_isa(bool has_sve) {
    if (has_sve) {
        if (mayiuse(sve_512)) {
            return cpu_isa_t::sve_512;
        }
        if (mayiuse(sve_256)) {
            return cpu_isa_t::sve_256;
        }
        if (mayiuse(sve_128)) {
            return cpu_isa_t::sve_128;
        }
    }
    return cpu_isa_t::asimd;
}

bool brgemm_int8_debug_enabled() {
    static const bool enabled = std::getenv("OV_CPU_JIT_INT8_CONV_DEBUG") != nullptr;
    return enabled;
}

template <typename... Ts>
void brgemm_int8_debug(Ts&&... args) {
    if (!brgemm_int8_debug_enabled()) {
        return;
    }
    std::ostringstream oss;
    (oss << ... << std::forward<Ts>(args));
    std::cerr << "[brgemm_int8_kernel] " << oss.str() << '\n';
}

const char* fallback_family_name(bool has_i8mm, bool has_dotprod) {
    if (has_i8mm) {
        return "aarch64_neon_i8mm";
    }
    if (has_dotprod) {
        return "aarch64_neon_dotprod";
    }
    return "reference_cpp";
}

const char* batch_kind_name(brgemm_batch_kind_t kind) {
    switch (kind) {
    case brgemm_addr:
        return "addr";
    case brgemm_offs:
        return "offs";
    case brgemm_strd:
        return "strd";
    case brgemm_static_offs:
        return "static_offs";
    default:
        return "unknown";
    }
}

const char* primary_fallback_kernel_name(bool src_signed, bool has_dotprod, bool has_i8mm, size_t M, size_t N) {
    if (M >= 4 && N >= 16 && has_i8mm) {
        return src_signed ? "aarch64_neon_i8mm_s8s8_4x16_packed" : "aarch64_neon_i8mm_u8s8_4x16_packed";
    }
    if (M >= 4 && N >= 8 && has_i8mm) {
        return src_signed ? "aarch64_neon_i8mm_s8s8_4x8_packed" : "aarch64_neon_i8mm_u8s8_4x8_packed";
    }
    if (M >= 4 && N >= 4 && has_i8mm) {
        return src_signed ? "aarch64_neon_i8mm_s8s8_4x4_packed" : "aarch64_neon_i8mm_u8s8_4x4_packed";
    }
    if (N >= 8 && has_dotprod) {
        if (M >= 2) {
            return src_signed ? "aarch64_neon_dotprod_s8s8_2x8" : "aarch64_neon_dotprod_u8s8_2x8";
        }
        return src_signed ? "aarch64_neon_dotprod_s8s8_1x8" : "aarch64_neon_dotprod_u8s8_1x8";
    }
    if (M >= 4 && N >= 4 && has_dotprod) {
        return src_signed ? "aarch64_neon_dotprod_s8s8_4x4" : "aarch64_neon_dotprod_u8s8_4x4";
    }
    if (N >= 4 && has_dotprod) {
        return src_signed ? "aarch64_neon_dotprod_s8s8_1x4" : "aarch64_neon_dotprod_u8s8_1x4";
    }
    return src_signed ? "aarch64_neon_mla_s8s8_1x4" : "aarch64_neon_mla_u8s8_1x4";
}

const char* selected_fallback_kernel_name(bool src_signed,
                                          bool can_use_dot,
                                          bool can_use_udot,
                                          bool can_use_i8mm,
                                          size_t m_block,
                                          size_t n_block) {
    if (m_block == 4 && n_block == 16 && can_use_i8mm) {
        return src_signed ? "aarch64_neon_i8mm_s8s8_4x16_packed" : "aarch64_neon_i8mm_u8s8_4x16_packed";
    }
    if (m_block == 4 && n_block == 8 && can_use_i8mm) {
        return src_signed ? "aarch64_neon_i8mm_s8s8_4x8_packed" : "aarch64_neon_i8mm_u8s8_4x8_packed";
    }
    if (m_block == 4 && n_block == 4 && can_use_i8mm) {
        return src_signed ? "aarch64_neon_i8mm_s8s8_4x4_packed" : "aarch64_neon_i8mm_u8s8_4x4_packed";
    }
    if (m_block == 4 && n_block == 4 && (can_use_dot || can_use_udot)) {
        return src_signed ? "aarch64_neon_dotprod_s8s8_4x4" : "aarch64_neon_dotprod_u8s8_4x4";
    }
    if (n_block == 8 && (can_use_dot || can_use_udot)) {
        if (m_block >= 2) {
            return src_signed ? "aarch64_neon_dotprod_s8s8_2x8" : "aarch64_neon_dotprod_u8s8_2x8";
        }
        return src_signed ? "aarch64_neon_dotprod_s8s8_1x8" : "aarch64_neon_dotprod_u8s8_1x8";
    }
    if (can_use_dot || can_use_udot) {
        return src_signed ? "aarch64_neon_dotprod_s8s8_1x4" : "aarch64_neon_dotprod_u8s8_1x4";
    }
    return src_signed ? "aarch64_neon_mla_s8s8_1x4" : "aarch64_neon_mla_u8s8_1x4";
}

size_t round_up(size_t value, size_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

size_t packed_block_stride_mmla(size_t K, size_t oc_block) {
    const size_t Kp = round_up(K, 8);
    return (Kp / 8) * oc_block * 8;
}

void pack_mmla_block(const int8_t* src, size_t K, size_t Kp, size_t oc_block, int8_t* dst) {
    const size_t k_blocks = Kp / 8;
    size_t offset = 0;
    for (size_t kb = 0; kb < k_blocks; ++kb) {
        const size_t k_off = kb * 8;
        for (size_t oc_pair = 0; oc_pair < oc_block; oc_pair += 2) {
            const size_t oc0 = oc_pair;
            const size_t oc1 = oc_pair + 1;
            for (size_t t = 0; t < 8; ++t) {
                const size_t k = k_off + t;
                const size_t dst_idx0 = offset + t;
                const size_t dst_idx1 = offset + 8 + t;
                dst[dst_idx0] = (oc0 < oc_block && k < K) ? src[oc0 * K + k] : 0;
                dst[dst_idx1] = (oc1 < oc_block && k < K) ? src[oc1 * K + k] : 0;
            }
            offset += 16;
        }
    }
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
                                   bool src_signed,
                                   brgemm_batch_kind_t batch_kind,
                                   const brgemm_batch_element_t* static_offsets,
                                   int max_bs) {
    M_ = M;
    N_ = N;
    K_ = K;
    lda_ = lda;
    ldb_ = ldb;
    ldb_fallback_ = K;
    ldc_ = ldc;
    src_signed_ = src_signed;
    batch_kind_ = batch_kind;
    if (static_offsets && max_bs > 0) {
        static_offsets_.assign(static_offsets, static_offsets + max_bs);
    }
    runtime_isa_ = ov::intel_cpu::getAarch64Int8Isa();
    has_sve_ = ov::intel_cpu::isSVEInt8Isa(runtime_isa_);
    has_dotprod_ = ov::intel_cpu::hasIntDotProductSupport();
    has_i8mm_ = ov::intel_cpu::hasInt8MMSupport();

    const auto dt_a = src_signed ? data_type::s8 : data_type::u8;
    const auto dt_b = data_type::s8;

    if (has_sve_) {
        const auto isa = get_preferred_brgemm_isa(has_sve_);
        brgemm_t brg = {};
        auto st = brgemm_desc_init(&brg,
                                   isa,
                                   batch_kind_,
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
        if (st == status::success && batch_kind_ != brgemm_addr && max_bs > 0) {
            brgemm_attr_t brgattr;
            brgattr.max_bs = max_bs;
            if (batch_kind_ == brgemm_static_offs && !static_offsets_.empty()) {
                brgattr.static_offsets = static_offsets_.data();
            }
            st = brgemm_desc_set_attr(&brg, brgattr);
        }
        if (st == status::success) {
            st = brgemm_desc_finalize(&brg);
        }
        if (st == status::success) {
            brgemm_kernel_t* kernel = nullptr;
            st = brgemm_kernel_create(&kernel, brg);
            if (st == status::success && kernel) {
                kernel_.reset(kernel);
                use_brgemm_ = true;
                brgemm_int8_debug("runtime_isa=", ov::intel_cpu::aarch64Int8IsaName(runtime_isa_),
                                  " family=onednn_brgemm_sve",
                                  " batch_kind=", batch_kind_name(batch_kind_),
                                  " M=", M_, " N=", N_, " K=", K_);
                return;
            }
        }
    }

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
    if (has_i8mm_) {
        if (src_signed_) {
            kernel_4x4_smmla_packed_ = std::make_unique<jit_int8_brgemm_kernel_4x4_smmla_packed>();
            kernel_4x4_smmla_packed_->create_ker();
            kernel_4x8_smmla_packed_ = std::make_unique<jit_int8_brgemm_kernel_4x8_smmla_packed>();
            kernel_4x8_smmla_packed_->create_ker();
            kernel_4x16_smmla_packed_ = std::make_unique<jit_int8_brgemm_kernel_4x16_smmla_packed>();
            kernel_4x16_smmla_packed_->create_ker();
        } else {
            kernel_4x4_usmmla_packed_ = std::make_unique<jit_int8_brgemm_kernel_4x4_usmmla_packed>();
            kernel_4x4_usmmla_packed_->create_ker();
            kernel_4x8_usmmla_packed_ = std::make_unique<jit_int8_brgemm_kernel_4x8_usmmla_packed>();
            kernel_4x8_usmmla_packed_->create_ker();
            kernel_4x16_usmmla_packed_ = std::make_unique<jit_int8_brgemm_kernel_4x16_usmmla_packed>();
            kernel_4x16_usmmla_packed_->create_ker();
        }
    }
    brgemm_int8_debug("runtime_isa=", ov::intel_cpu::aarch64Int8IsaName(runtime_isa_),
                      " family=", fallback_family_name(has_i8mm_, has_dotprod_),
                      " primary_kernel=", primary_fallback_kernel_name(src_signed_, has_dotprod_, has_i8mm_, M_, N_),
                      " batch_kind=", batch_kind_name(batch_kind_),
                      " M=", M_, " N=", N_, " K=", K_);
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
    if (batch_kind_ != brgemm_addr) {
        return;
    }
    if (use_brgemm_) {
        brgemm_kernel_execute(kernel_.get(), bs, batch, c, nullptr);
        return;
    }
    execute_fallback(batch, bs, c);
}

void BrgemmInt8Kernel::execute_batch_offsets(const void* base_a,
                                             const int8_t* base_b,
                                             const brgemm_batch_element_t* batch,
                                             int bs,
                                             int32_t* c) const {
    const auto* offsets = batch ? batch : static_offsets_.data();
    if (!offsets || bs <= 0) {
        return;
    }
    if (use_brgemm_) {
        brgemm_kernel_execute(kernel_.get(), bs, base_a, base_b, offsets, c, nullptr);
        return;
    }
    execute_fallback_offsets(base_a, base_b, offsets, bs, c);
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
    const bool can_use_i8mm4_s8 = src_signed_ && has_i8mm_ && kernel_4x4_smmla_packed_;
    const bool can_use_i8mm8_s8 = src_signed_ && has_i8mm_ && kernel_4x8_smmla_packed_;
    const bool can_use_i8mm16_s8 = src_signed_ && has_i8mm_ && kernel_4x16_smmla_packed_;
    const bool can_use_i8mm4_u8 = !src_signed_ && has_i8mm_ && kernel_4x4_usmmla_packed_;
    const bool can_use_i8mm8_u8 = !src_signed_ && has_i8mm_ && kernel_4x8_usmmla_packed_;
    const bool can_use_i8mm16_u8 = !src_signed_ && has_i8mm_ && kernel_4x16_usmmla_packed_;
    const size_t k_padded = round_up(K, 8);
    const size_t packed_stride4 = packed_block_stride_mmla(K, 4);
    const size_t packed_stride8 = packed_block_stride_mmla(K, 8);
    const size_t packed_stride16 = packed_block_stride_mmla(K, 16);
    std::vector<int8_t> packed_wei(std::max({packed_stride4, packed_stride8, packed_stride16}));
    const auto log_selected_kernel = [&](const char* kernel_name) {
        bool expected = false;
        if (logged_execute_path_.compare_exchange_strong(expected, true)) {
            brgemm_int8_debug("selected_kernel=",
                              kernel_name,
                              " family=", fallback_family_name(has_i8mm_, has_dotprod_),
                              " batch_kind=", batch_kind_name(batch_kind_),
                              " M=", M_, " N=", N_, " K=", K_, " bs=", bs);
        }
    };

    for (size_t m0 = 0; m0 < M; m0 += 4) {
        const size_t m_block = std::min<size_t>(4, M - m0);
        size_t n0 = 0;
        while (n0 < N) {
            size_t n_step = 4;
            if (m_block == 4) {
                if ((can_use_i8mm16_s8 || can_use_i8mm16_u8) && (N - n0) >= 16) {
                    n_step = 16;
                } else if ((can_use_i8mm8_s8 || can_use_i8mm8_u8 || can_use_dot8 || can_use_udot8) && (N - n0) >= 8) {
                    n_step = 8;
                }
            } else if ((can_use_dot8 || can_use_udot8) && (N - n0) >= 8) {
                n_step = 8;
            }
            const size_t n_block = std::min<size_t>(n_step, N - n0);

            if (m_block == 4 && n_block == 16 && (can_use_i8mm16_s8 || can_use_i8mm16_u8)) {
                log_selected_kernel(selected_fallback_kernel_name(src_signed_, false, false, true, m_block, n_block));
                for (int b = 0; b < bs; ++b) {
                    const auto* wei = static_cast<const int8_t*>(batch[b].ptr.B);
                    pack_mmla_block(wei + n0 * ldb, K, k_padded, 16, packed_wei.data());
                    const int accum = b == 0 ? 0 : 1;
                    int32_t* dst = c + m0 * ldc + n0;
                    if (src_signed_) {
                        const auto* src = static_cast<const int8_t*>(batch[b].ptr.A);
                        const int8_t* srcs[4] = {
                            src + (m0 + 0) * lda,
                            src + (m0 + 1) * lda,
                            src + (m0 + 2) * lda,
                            src + (m0 + 3) * lda,
                        };
                        kernel_4x16_smmla_packed_->ker()(srcs,
                                                         packed_wei.data(),
                                                         dst,
                                                         K,
                                                         0,
                                                         ldc * sizeof(int32_t),
                                                         accum);
                    } else {
                        const auto* src = static_cast<const uint8_t*>(batch[b].ptr.A);
                        const uint8_t* srcs[4] = {
                            src + (m0 + 0) * lda,
                            src + (m0 + 1) * lda,
                            src + (m0 + 2) * lda,
                            src + (m0 + 3) * lda,
                        };
                        kernel_4x16_usmmla_packed_->ker()(srcs,
                                                          packed_wei.data(),
                                                          dst,
                                                          K,
                                                          0,
                                                          ldc * sizeof(int32_t),
                                                          accum);
                    }
                }
                n0 += n_block;
                continue;
            }

            if (m_block == 4 && n_block == 8 && (can_use_i8mm8_s8 || can_use_i8mm8_u8)) {
                log_selected_kernel(selected_fallback_kernel_name(src_signed_, false, false, true, m_block, n_block));
                for (int b = 0; b < bs; ++b) {
                    const auto* wei = static_cast<const int8_t*>(batch[b].ptr.B);
                    pack_mmla_block(wei + n0 * ldb, K, k_padded, 8, packed_wei.data());
                    const int accum = b == 0 ? 0 : 1;
                    int32_t* dst = c + m0 * ldc + n0;
                    if (src_signed_) {
                        const auto* src = static_cast<const int8_t*>(batch[b].ptr.A);
                        const int8_t* srcs[4] = {
                            src + (m0 + 0) * lda,
                            src + (m0 + 1) * lda,
                            src + (m0 + 2) * lda,
                            src + (m0 + 3) * lda,
                        };
                        kernel_4x8_smmla_packed_->ker()(srcs,
                                                        packed_wei.data(),
                                                        dst,
                                                        K,
                                                        0,
                                                        ldc * sizeof(int32_t),
                                                        accum);
                    } else {
                        const auto* src = static_cast<const uint8_t*>(batch[b].ptr.A);
                        const uint8_t* srcs[4] = {
                            src + (m0 + 0) * lda,
                            src + (m0 + 1) * lda,
                            src + (m0 + 2) * lda,
                            src + (m0 + 3) * lda,
                        };
                        kernel_4x8_usmmla_packed_->ker()(srcs,
                                                         packed_wei.data(),
                                                         dst,
                                                         K,
                                                         nullptr,
                                                         ldc * sizeof(int32_t),
                                                         accum);
                    }
                }
                n0 += n_block;
                continue;
            }

            if (m_block == 4 && n_block == 4) {
                if (can_use_i8mm4_s8 || can_use_i8mm4_u8) {
                    log_selected_kernel(selected_fallback_kernel_name(src_signed_, false, false, true, m_block, n_block));
                    for (int b = 0; b < bs; ++b) {
                        const auto* wei = static_cast<const int8_t*>(batch[b].ptr.B);
                        pack_mmla_block(wei + n0 * ldb, K, k_padded, 4, packed_wei.data());
                        const int accum = b == 0 ? 0 : 1;
                        int32_t* dst = c + m0 * ldc + n0;
                        if (src_signed_) {
                            const auto* src = static_cast<const int8_t*>(batch[b].ptr.A);
                            const int8_t* srcs[4] = {
                                src + (m0 + 0) * lda,
                                src + (m0 + 1) * lda,
                                src + (m0 + 2) * lda,
                                src + (m0 + 3) * lda,
                            };
                            kernel_4x4_smmla_packed_->ker()(srcs,
                                                            packed_wei.data(),
                                                            dst,
                                                            K,
                                                            0,
                                                            ldc * sizeof(int32_t),
                                                            accum);
                        } else {
                            const auto* src = static_cast<const uint8_t*>(batch[b].ptr.A);
                            const uint8_t* srcs[4] = {
                                src + (m0 + 0) * lda,
                                src + (m0 + 1) * lda,
                                src + (m0 + 2) * lda,
                                src + (m0 + 3) * lda,
                            };
                            kernel_4x4_usmmla_packed_->ker()(srcs,
                                                             packed_wei.data(),
                                                             dst,
                                                             K,
                                                             nullptr,
                                                             ldc * sizeof(int32_t),
                                                             accum);
                        }
                    }
                    n0 += n_block;
                    continue;
                }
                if (can_use_dot) {
                    log_selected_kernel(selected_fallback_kernel_name(true, true, false, false, m_block, n_block));
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
                    log_selected_kernel(selected_fallback_kernel_name(false, false, true, false, m_block, n_block));
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
                log_selected_kernel(selected_fallback_kernel_name(true, true, false, false, m_block, n_block));
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
                log_selected_kernel(selected_fallback_kernel_name(false, false, true, false, m_block, n_block));
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
                log_selected_kernel(
                    selected_fallback_kernel_name(src_signed_, src_signed_ && has_dotprod_, !src_signed_ && can_use_udot, false, m_block, n_block));
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

void BrgemmInt8Kernel::execute_fallback_offsets(const void* base_a,
                                                const int8_t* base_b,
                                                const brgemm_batch_element_t* batch,
                                                int bs,
                                                int32_t* c) const {
    if (!batch || bs <= 0) {
        return;
    }
    std::vector<brgemm_batch_element_t> addr_batch(static_cast<size_t>(bs));
    for (int i = 0; i < bs; ++i) {
        addr_batch[i].ptr.A = static_cast<const uint8_t*>(base_a) + batch[i].offset.A;
        addr_batch[i].ptr.B = reinterpret_cast<const int8_t*>(base_b) + batch[i].offset.B;
        addr_batch[i].vvpad.top = batch[i].vvpad.top;
        addr_batch[i].vvpad.bottom = batch[i].vvpad.bottom;
    }
    execute_fallback(addr_batch.data(), bs, c);
}

}  // namespace ov::intel_cpu::aarch64
