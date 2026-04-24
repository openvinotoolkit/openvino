// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>

#include <cpu/aarch64/jit_generator.hpp>

namespace ov::intel_cpu::aarch64 {

class jit_int8_brgemm_kernel_1x4 : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_1x4)

    using ker_t = void (*)(const uint8_t* src, const int8_t* wei, int32_t* dst, size_t K, size_t ldB, size_t accum);

    explicit jit_int8_brgemm_kernel_1x4(bool src_signed);

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    bool src_signed_ = false;
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_1x4_dot : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_1x4_dot)

    using ker_t = void (*)(const int8_t* src, const int8_t* wei, int32_t* dst, size_t K, size_t ldB, size_t accum);

    jit_int8_brgemm_kernel_1x4_dot();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_1x4_udot : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_1x4_udot)

    using ker_t = void (*)(const uint8_t* src, const int8_t* wei, int32_t* dst, size_t K, size_t ldB, size_t accum);

    jit_int8_brgemm_kernel_1x4_udot();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_1x8_dot : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_1x8_dot)

    using ker_t = void (*)(const int8_t* src, const int8_t* wei, int32_t* dst, size_t K, size_t ldB, size_t accum);

    jit_int8_brgemm_kernel_1x8_dot();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_1x8_udot : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_1x8_udot)

    using ker_t = void (*)(const uint8_t* src, const int8_t* wei, int32_t* dst, size_t K, size_t ldB, size_t accum);

    jit_int8_brgemm_kernel_1x8_udot();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_4x4_dot : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_4x4_dot)

    using ker_t = void (*)(const int8_t* const* srcs,
                           const int8_t* wei,
                           int32_t* dst,
                           size_t K,
                           size_t ldB,
                           size_t ldC,
                           size_t accum);

    jit_int8_brgemm_kernel_4x4_dot();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_4x4_udot : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_4x4_udot)

    using ker_t = void (*)(const uint8_t* const* srcs,
                           const int8_t* wei,
                           int32_t* dst,
                           size_t K,
                           size_t ldB,
                           size_t ldC,
                           size_t accum);

    jit_int8_brgemm_kernel_4x4_udot();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_2x8_dot : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_2x8_dot)

    using ker_t = void (*)(const int8_t* const* srcs,
                           const int8_t* wei,
                           int32_t* dst,
                           size_t K,
                           size_t ldB,
                           size_t ldC,
                           size_t accum);

    jit_int8_brgemm_kernel_2x8_dot();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_2x8_udot : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_2x8_udot)

    using ker_t = void (*)(const uint8_t* const* srcs,
                           const int8_t* wei,
                           int32_t* dst,
                           size_t K,
                           size_t ldB,
                           size_t ldC,
                           size_t accum);

    jit_int8_brgemm_kernel_2x8_udot();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

}  // namespace ov::intel_cpu::aarch64
