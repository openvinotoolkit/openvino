// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace dnnl::impl::cpu::aarch64 {
struct brgemm_batch_element_t;
struct brgemm_kernel_t;
}

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

class BrgemmInt8Kernel {
public:
    BrgemmInt8Kernel(size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc, bool src_signed);
    ~BrgemmInt8Kernel();

    void execute(const void* a, const int8_t* b, int32_t* c) const;
    void execute_batch(const dnnl::impl::cpu::aarch64::brgemm_batch_element_t* batch, int bs, int32_t* c) const;
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
    bool use_brgemm_ = false;
    bool src_signed_ = false;
    bool has_dotprod_ = false;
    size_t M_ = 0;
    size_t N_ = 0;
    size_t K_ = 0;
    size_t lda_ = 0;
    size_t ldb_ = 0;
    size_t ldb_fallback_ = 0;
    size_t ldc_ = 0;

    void execute_fallback(const dnnl::impl::cpu::aarch64::brgemm_batch_element_t* batch,
                          int bs,
                          int32_t* c) const;
};

}  // namespace ov::intel_cpu::aarch64
