// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>

#include <cpu/aarch64/jit_generator.hpp>

namespace ov::intel_cpu::aarch64 {

class jit_int8_brgemm_kernel_1x8_dot_packed : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_1x8_dot_packed)

    using ker_t = void (*)(const int8_t* src, const int8_t* wei, int32_t* dst, size_t K, size_t ldB, size_t accum);

    jit_int8_brgemm_kernel_1x8_dot_packed();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_1x8_udot_packed : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_1x8_udot_packed)

    using ker_t = void (*)(const uint8_t* src, const int8_t* wei, int32_t* dst, size_t K, size_t ldB, size_t accum);

    jit_int8_brgemm_kernel_1x8_udot_packed();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_2x8_dot_packed : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_2x8_dot_packed)

    using ker_t = void (*)(const int8_t* const* srcs,
                           const int8_t* wei,
                           int32_t* dst,
                           size_t K,
                           size_t ldB,
                           size_t ldC,
                           size_t accum);

    jit_int8_brgemm_kernel_2x8_dot_packed();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_2x8_dot_packed_strided : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_2x8_dot_packed_strided)

    using ker_t = void (*)(const int8_t* src,
                           const int8_t* wei,
                           int32_t* dst,
                           size_t K,
                           size_t src_stride,
                           size_t ldC,
                           size_t accum);

    jit_int8_brgemm_kernel_2x8_dot_packed_strided();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_2x8_dot_packed_strided_interleaved4 : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_2x8_dot_packed_strided_interleaved4)

    using ker_t = void (*)(const int8_t* src,
                           const int8_t* wei,
                           int32_t* dst,
                           size_t K,
                           size_t src_stride,
                           size_t ldC,
                           size_t accum);

    jit_int8_brgemm_kernel_2x8_dot_packed_strided_interleaved4();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_2x16_dot_packed_strided_interleaved4 : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_2x16_dot_packed_strided_interleaved4)

    using ker_t = void (*)(const int8_t* src,
                           const int8_t* wei,
                           int32_t* dst,
                           size_t K,
                           size_t src_stride,
                           size_t ldC,
                           size_t accum);

    jit_int8_brgemm_kernel_2x16_dot_packed_strided_interleaved4();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_2x32_dot_packed_strided_interleaved4 : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_2x32_dot_packed_strided_interleaved4)

    using ker_t = void (*)(const int8_t* src,
                           const int8_t* wei,
                           int32_t* dst,
                           size_t K,
                           size_t src_stride,
                           size_t ldC,
                           size_t accum);

    jit_int8_brgemm_kernel_2x32_dot_packed_strided_interleaved4();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_4x16_dot_packed_strided_interleaved4 : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_4x16_dot_packed_strided_interleaved4)

    using ker_t = void (*)(const int8_t* src,
                           const int8_t* wei,
                           int32_t* dst,
                           size_t K,
                           size_t src_stride,
                           size_t ldC,
                           size_t accum);

    jit_int8_brgemm_kernel_4x16_dot_packed_strided_interleaved4();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_4x16_dot_packed_lhs_strided_interleaved4
    : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_4x16_dot_packed_lhs_strided_interleaved4)

    using ker_t = void (*)(const int8_t* src,
                           const int8_t* wei,
                           int32_t* dst,
                           size_t K,
                           size_t src_stride,
                           size_t ldC,
                           size_t accum);

    jit_int8_brgemm_kernel_4x16_dot_packed_lhs_strided_interleaved4();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_2x8_udot_packed : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_2x8_udot_packed)

    using ker_t = void (*)(const uint8_t* const* srcs,
                           const int8_t* wei,
                           int32_t* dst,
                           size_t K,
                           size_t ldB,
                           size_t ldC,
                           size_t accum);

    jit_int8_brgemm_kernel_2x8_udot_packed();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_2x8_udot_packed_strided : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_2x8_udot_packed_strided)

    using ker_t = void (*)(const uint8_t* src,
                           const int8_t* wei,
                           int32_t* dst,
                           size_t K,
                           size_t src_stride,
                           size_t ldC,
                           size_t accum);

    jit_int8_brgemm_kernel_2x8_udot_packed_strided();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_2x8_udot_packed_strided_interleaved4 : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_2x8_udot_packed_strided_interleaved4)

    using ker_t = void (*)(const uint8_t* src,
                           const int8_t* wei,
                           int32_t* dst,
                           size_t K,
                           size_t src_stride,
                           size_t ldC,
                           size_t accum);

    jit_int8_brgemm_kernel_2x8_udot_packed_strided_interleaved4();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_2x16_udot_packed_strided_interleaved4 : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_2x16_udot_packed_strided_interleaved4)

    using ker_t = void (*)(const uint8_t* src,
                           const int8_t* wei,
                           int32_t* dst,
                           size_t K,
                           size_t src_stride,
                           size_t ldC,
                           size_t accum);

    jit_int8_brgemm_kernel_2x16_udot_packed_strided_interleaved4();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_2x32_udot_packed_strided_interleaved4 : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_2x32_udot_packed_strided_interleaved4)

    using ker_t = void (*)(const uint8_t* src,
                           const int8_t* wei,
                           int32_t* dst,
                           size_t K,
                           size_t src_stride,
                           size_t ldC,
                           const int32_t* bias,
                           size_t accum);

    jit_int8_brgemm_kernel_2x32_udot_packed_strided_interleaved4();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_4x16_udot_packed_strided_interleaved4 : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_4x16_udot_packed_strided_interleaved4)

    using ker_t = void (*)(const uint8_t* src,
                           const int8_t* wei,
                           int32_t* dst,
                           size_t K,
                           size_t src_stride,
                           size_t ldC,
                           size_t accum);

    jit_int8_brgemm_kernel_4x16_udot_packed_strided_interleaved4();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_4x16_udot_packed_lhs_strided_interleaved4
    : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_4x16_udot_packed_lhs_strided_interleaved4)

    using ker_t = void (*)(const uint8_t* src,
                           const int8_t* wei,
                           int32_t* dst,
                           size_t K,
                           size_t src_stride,
                           size_t ldC,
                           size_t accum);

    jit_int8_brgemm_kernel_4x16_udot_packed_lhs_strided_interleaved4();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_4x4_dot_packed : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_4x4_dot_packed)

    using ker_t = void (*)(const int8_t* const* srcs,
                           const int8_t* wei,
                           int32_t* dst,
                           size_t K,
                           size_t ldB,
                           size_t ldC,
                           size_t accum);

    jit_int8_brgemm_kernel_4x4_dot_packed();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

class jit_int8_brgemm_kernel_4x4_udot_packed : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_brgemm_kernel_4x4_udot_packed)

    using ker_t = void (*)(const uint8_t* const* srcs,
                           const int8_t* wei,
                           int32_t* dst,
                           size_t K,
                           size_t ldB,
                           size_t ldC,
                           size_t accum);

    jit_int8_brgemm_kernel_4x4_udot_packed();

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    ker_t ker_ = nullptr;
};

}  // namespace ov::intel_cpu::aarch64
