// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <xbyak/xbyak.h>

#include <cassert>
#include <common/utils.hpp>
#include <cstddef>
#include <memory>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/jit_uni_depthwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_quantization_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/cpu_utils.hpp"

namespace ov::intel_cpu {

using namespace dnnl::impl::cpu::x64;

struct jit_matmul_small_config_params {
    size_t M = 0UL;
    size_t K = 0UL;
    size_t N = 0UL;
};

struct jit_matmul_small_call_args {
    const void* input1;
    const void* input2;
    const void* post_op_data;
    void* output;
    size_t oc;
    size_t oc_off;
    size_t B;
};

struct jit_uni_matmul_small_kernel {
    void (*ker_)(const jit_matmul_small_call_args*) = nullptr;

    void operator()(const jit_matmul_small_call_args* args) const {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_matmul_small_kernel(jit_matmul_small_config_params jcp, const dnnl_primitive_attr& attr)
        : jcp_(jcp),
          attr_(attr) {}
    virtual ~jit_uni_matmul_small_kernel() = default;

    virtual void create_ker() = 0;

    jit_matmul_small_config_params jcp_;
    const dnnl_primitive_attr& attr_;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
struct jit_uni_matmul_small_kernel_f32 : public jit_uni_matmul_small_kernel,
                                         public dnnl::impl::cpu::x64::jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_matmul_small_kernel_f32)

    explicit jit_uni_matmul_small_kernel_f32(jit_matmul_small_config_params jcp, const dnnl_primitive_attr& attr)
        : jit_uni_matmul_small_kernel(jcp, attr),
          jit_generator_t(jit_name()) {}

    void create_ker() override {
        jit_generator_t::create_kernel();
        ker_ = jit_kernel_cast<decltype(ker_)>(jit_ker());
    }

    void generate() override;

private:
    void apply_post_ops(ov::element::Type dst_prc, size_t vmm_idx, bool is_broadcast);
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                                         Xbyak::Xmm,
                                                         isa == dnnl::impl::cpu::x64::avx2,
                                                         Xbyak::Ymm,
                                                         Xbyak::Zmm>::type;

    const int vlen = dnnl::impl::cpu::x64::cpu_isa_traits_t<isa>::vlen;

    Xbyak::Reg64 reg_input1 = r8;
    Xbyak::Reg64 reg_input2 = r9;
    Xbyak::Reg64 reg_post_ops_data = rbx;
    Xbyak::Reg64 reg_out = r10;
    Xbyak::Reg64 reg_work_amount = r11;
    Xbyak::Reg64 reg_oc = r12;
    Xbyak::Reg64 reg_oc_off = r13;
    Xbyak::Reg64 reg_params = Xbyak::Reg64(dnnl::impl::cpu::x64::abi_param_regs[0]);
    Xbyak::Reg64 reg_d_weights = r14;
    Xbyak::Reg64 reg_d_bias = r15;

    Vmm vmm_input1[4] = {Vmm(0), Vmm(1), Vmm(2), Vmm(3)};
    Vmm vmm_input2[4] = {Vmm(4), Vmm(5), Vmm(6), Vmm(7)};
    Vmm vmm_output[4] = {Vmm(8), Vmm(9), Vmm(10), Vmm(11)};
    Vmm vmm_d_weights = Vmm(12);
    Vmm vmm_d_bias = Vmm(13);

    std::vector<std::shared_ptr<jit_uni_eltwise_injector_t<isa>>> eltwise_injectors;
    std::vector<std::shared_ptr<jit_uni_depthwise_injector_f32<isa>>> depthwise_injectors;
    std::vector<std::shared_ptr<jit_uni_quantization_injector_f32<isa>>> quantization_injectors;
};

}  // namespace ov::intel_cpu
