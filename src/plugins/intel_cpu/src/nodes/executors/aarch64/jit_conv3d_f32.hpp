// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <cpu/aarch64/jit_generator.hpp>

namespace ov::intel_cpu {

struct jit_conv3d_f32_call_args {
    const float* src;
    const float* wei;
    const float* wei2;
    size_t repeats;
    size_t tail;
    size_t src_stride;
    size_t wei_stride;
    size_t src_blk_stride;
    size_t wei_blk_stride;
    float* acc;
    float* acc2;
    size_t kw_cnt;
    size_t src_dx;
    size_t wei_dx;
};

class JitConv3DKernelF32 : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(JitConv3DKernelF32)
    using jit_fn = void (*)(const jit_conv3d_f32_call_args*);

    JitConv3DKernelF32() = default;

    void create_ker();
    inline void operator()(const jit_conv3d_f32_call_args* p) const {
        ker_(p);
    }

private:
    void generate() override;

    jit_fn ker_{nullptr};
};

}  // namespace ov::intel_cpu
