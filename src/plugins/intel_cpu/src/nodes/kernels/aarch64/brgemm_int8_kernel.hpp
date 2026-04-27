// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <cpu/aarch64/brgemm/brgemm_types.hpp>
#include "utils/precision_support.h"

namespace ov::intel_cpu::aarch64 {

class jit_int8_brgemm_kernel_1x4;
class jit_int8_brgemm_kernel_1x4_dot;
class jit_int8_brgemm_kernel_1x4_udot;
class jit_int8_brgemm_kernel_1x8_dot;
class jit_int8_brgemm_kernel_1x8_udot;
class jit_int8_brgemm_kernel_2x8_dot;
class jit_int8_brgemm_kernel_2x8_udot;
class jit_int8_brgemm_kernel_4x4_dot;
class jit_int8_brgemm_kernel_4x4_udot;
class jit_int8_brgemm_kernel_4x4_smmla_packed;
class jit_int8_brgemm_kernel_4x8_smmla_packed;
class jit_int8_brgemm_kernel_4x16_smmla_packed;
class jit_int8_brgemm_kernel_4x4_usmmla_packed;
class jit_int8_brgemm_kernel_4x8_usmmla_packed;
class jit_int8_brgemm_kernel_4x16_usmmla_packed;

class BrgemmInt8Kernel {
public:
    BrgemmInt8Kernel(size_t M,
                     size_t N,
                     size_t K,
                     size_t lda,
                     size_t ldb,
                     size_t ldc,
                     bool src_signed,
                     dnnl::impl::cpu::aarch64::brgemm_batch_kind_t batch_kind =
                         dnnl::impl::cpu::aarch64::brgemm_addr,
                     const dnnl::impl::cpu::aarch64::brgemm_batch_element_t* static_offsets = nullptr,
                     int max_bs = 0);
    ~BrgemmInt8Kernel();

    void execute(const void* a, const int8_t* b, int32_t* c) const;
    void execute_batch(const dnnl::impl::cpu::aarch64::brgemm_batch_element_t* batch, int bs, int32_t* c) const;
    void execute_batch_offsets(const void* base_a,
                               const int8_t* base_b,
                               const dnnl::impl::cpu::aarch64::brgemm_batch_element_t* batch,
                               int bs,
                               int32_t* c) const;
    [[nodiscard]] bool uses_brgemm() const noexcept {
        return use_brgemm_;
    }

private:
    struct BrgKernelDeleter {
        void operator()(dnnl::impl::cpu::aarch64::brgemm_kernel_t* ptr) const noexcept;
    };

    std::unique_ptr<dnnl::impl::cpu::aarch64::brgemm_kernel_t, BrgKernelDeleter> kernel_;
    std::unique_ptr<jit_int8_brgemm_kernel_1x4> kernel_1x4_;
    std::unique_ptr<jit_int8_brgemm_kernel_1x4_dot> kernel_1x4_dot_;
    std::unique_ptr<jit_int8_brgemm_kernel_1x4_udot> kernel_1x4_udot_;
    std::unique_ptr<jit_int8_brgemm_kernel_1x8_dot> kernel_1x8_dot_;
    std::unique_ptr<jit_int8_brgemm_kernel_1x8_udot> kernel_1x8_udot_;
    std::unique_ptr<jit_int8_brgemm_kernel_2x8_dot> kernel_2x8_dot_;
    std::unique_ptr<jit_int8_brgemm_kernel_2x8_udot> kernel_2x8_udot_;
    std::unique_ptr<jit_int8_brgemm_kernel_4x4_dot> kernel_4x4_dot_;
    std::unique_ptr<jit_int8_brgemm_kernel_4x4_udot> kernel_4x4_udot_;
    std::unique_ptr<jit_int8_brgemm_kernel_4x4_smmla_packed> kernel_4x4_smmla_packed_;
    std::unique_ptr<jit_int8_brgemm_kernel_4x8_smmla_packed> kernel_4x8_smmla_packed_;
    std::unique_ptr<jit_int8_brgemm_kernel_4x16_smmla_packed> kernel_4x16_smmla_packed_;
    std::unique_ptr<jit_int8_brgemm_kernel_4x4_usmmla_packed> kernel_4x4_usmmla_packed_;
    std::unique_ptr<jit_int8_brgemm_kernel_4x8_usmmla_packed> kernel_4x8_usmmla_packed_;
    std::unique_ptr<jit_int8_brgemm_kernel_4x16_usmmla_packed> kernel_4x16_usmmla_packed_;
    bool use_brgemm_ = false;
    bool src_signed_ = false;
    bool has_dotprod_ = false;
    bool has_i8mm_ = false;
    bool has_sve_ = false;
    ov::intel_cpu::Aarch64Int8Isa runtime_isa_ = ov::intel_cpu::Aarch64Int8Isa::scalar_reference;
    dnnl::impl::cpu::aarch64::brgemm_batch_kind_t batch_kind_ = dnnl::impl::cpu::aarch64::brgemm_addr;
    size_t M_ = 0;
    size_t N_ = 0;
    size_t K_ = 0;
    size_t lda_ = 0;
    size_t ldb_ = 0;
    size_t ldb_fallback_ = 0;
    size_t ldc_ = 0;
    std::vector<dnnl::impl::cpu::aarch64::brgemm_batch_element_t> static_offsets_;
    mutable std::atomic<bool> logged_execute_path_{false};

    void execute_fallback(const dnnl::impl::cpu::aarch64::brgemm_batch_element_t* batch,
                          int bs,
                          int32_t* c) const;
    void execute_fallback_offsets(const void* base_a,
                                  const int8_t* base_b,
                                  const dnnl::impl::cpu::aarch64::brgemm_batch_element_t* batch,
                                  int bs,
                                  int32_t* c) const;
};

}  // namespace ov::intel_cpu::aarch64
