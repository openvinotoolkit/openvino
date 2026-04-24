// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>

#include <cpu/aarch64/jit_generator.hpp>

namespace ov::intel_cpu::aarch64 {

class jit_int8_dot_kernel : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_dot_kernel)

    using ker_t = void (*)(const uint8_t* src, const int8_t* wei, int32_t* dst, size_t K, size_t accum);

    explicit jit_int8_dot_kernel(bool src_signed);

    void create_ker();

    ker_t ker() const {
        return ker_;
    }

    void generate() override;

private:
    bool src_signed_ = false;
    ker_t ker_ = nullptr;
};

}  // namespace ov::intel_cpu::aarch64
