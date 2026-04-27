// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>

#include <cpu/aarch64/jit_generator.hpp>

namespace ov::intel_cpu::aarch64 {

class jit_int8_brgemm_kernel_4x4_smmla_packed : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_4x4_smmla_packed)

    using ker_t = void (*)(const int8_t* const* srcs,
                           const int8_t* wei,
                           int32_t* dst,
                           size_t K,
                           size_t ldB,
                           size_t ldC,
                           size_t accum);

    explicit jit_int8_brgemm_kernel_4x4_smmla_packed(bool interleaved = false);

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    bool interleaved_ = false;
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_4x8_smmla_packed : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_4x8_smmla_packed)

    using ker_t = void (*)(const int8_t* const* srcs,
                           const int8_t* wei,
                           int32_t* dst,
                           size_t K,
                           size_t ldB,
                           size_t ldC,
                           size_t accum);

    explicit jit_int8_brgemm_kernel_4x8_smmla_packed(bool interleaved = false);

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    bool interleaved_ = false;
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_8x8_smmla_packed : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_8x8_smmla_packed)

    using ker_t = void (*)(const int8_t* const* srcs,
                           const int8_t* wei,
                           int32_t* dst,
                           size_t K,
                           size_t ldB,
                           size_t ldC,
                           size_t accum);

    jit_int8_brgemm_kernel_8x8_smmla_packed();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_8x12_smmla_packed : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_8x12_smmla_packed)

    using ker_t = void (*)(const int8_t* const* srcs,
                           const int8_t* wei,
                           int32_t* dst,
                           size_t K,
                           size_t ldB,
                           size_t ldC,
                           size_t accum);

    jit_int8_brgemm_kernel_8x12_smmla_packed();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_4x16_smmla_packed : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_4x16_smmla_packed)

    using ker_t = void (*)(const int8_t* const* srcs,
                           const int8_t* wei,
                           int32_t* dst,
                           size_t K,
                           size_t ldB,
                           size_t ldC,
                           size_t accum);

    explicit jit_int8_brgemm_kernel_4x16_smmla_packed(bool interleaved = false);

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    bool interleaved_ = false;
    ker_t ker_ = nullptr;
};

}  // namespace ov::intel_cpu::aarch64
