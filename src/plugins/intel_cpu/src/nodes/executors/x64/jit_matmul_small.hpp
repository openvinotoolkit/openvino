// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/x64/xbyak/xbyak.h>

#include <cassert>
#include <common/utils.hpp>
#include <cstddef>
#include <memory>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace ov::intel_cpu {

struct jit_matmul_small_config_params {
    size_t M;
    size_t K;
    size_t N;
};

struct jit_matmul_small_call_args {
    const void* input1;
    const void* input2;
    void* output;
    size_t B;
};

struct jit_uni_matmul_small_kernel {
    void (*ker_)(const jit_matmul_small_call_args*);

    void operator()(const jit_matmul_small_call_args* args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_matmul_small_kernel(jit_matmul_small_config_params jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_matmul_small_kernel() {}

    virtual void create_ker() = 0;

    jit_matmul_small_config_params jcp_;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
struct jit_uni_matmul_small_kernel_f32 : public jit_uni_matmul_small_kernel,
                                         public dnnl::impl::cpu::x64::jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_matmul_small_kernel_f32)

    explicit jit_uni_matmul_small_kernel_f32(jit_matmul_small_config_params jcp)
        : jit_uni_matmul_small_kernel(jcp),
          jit_generator(jit_name()) {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override;

private:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                                         Xbyak::Xmm,
                                                         isa == dnnl::impl::cpu::x64::avx2,
                                                         Xbyak::Ymm,
                                                         Xbyak::Zmm>::type;

    const int vlen = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 reg_input1 = r8;
    Xbyak::Reg64 reg_input2 = r9;
    Xbyak::Reg64 reg_out = r10;
    Xbyak::Reg64 reg_work_amount = r11;
    Xbyak::Reg64 reg_params = Xbyak::Reg64(dnnl::impl::cpu::x64::abi_param_regs[0]);

    Vmm vmm_input1[4] = {Vmm(0), Vmm(1), Vmm(2), Vmm(3)};
    Vmm vmm_input2[4] = {Vmm(4), Vmm(5), Vmm(6), Vmm(7)};
    Vmm vmm_output[4] = {Vmm(8), Vmm(9), Vmm(10), Vmm(11)};
};

}  // namespace ov::intel_cpu
