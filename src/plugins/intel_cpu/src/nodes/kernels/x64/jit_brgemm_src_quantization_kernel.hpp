// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <common/c_types_map.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>

namespace ov::intel_cpu {

struct src_quantization_compile_params_t {
    size_t ic_quant_block;
    bool with_src_grouped_sum;
    size_t src_sum_group_size;
    dnnl::impl::data_type_t src_dt;
    dnnl::impl::data_type_t qsrc_dt;
};

struct src_quantization_runtime_params_t {
    const void* src_ptr;
    const void* qsrc_ptr;
    const void* src_scales_ptr;
    const void* src_grouped_sum_ptr;
    size_t ic_size;
};

struct jit_src_quantization_kernel_base_t {
    void operator()(const src_quantization_runtime_params_t* args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_src_quantization_kernel_base_t(const src_quantization_compile_params_t& jcp) : jcp_(jcp) {}
    virtual ~jit_src_quantization_kernel_base_t() = default;

protected:
    void (*ker_)(const src_quantization_runtime_params_t*) = nullptr;
    src_quantization_compile_params_t jcp_;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
struct jit_brgemm_src_quantization_kernel_t : public jit_src_quantization_kernel_base_t,
                                              public dnnl::impl::cpu::x64::jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_src_quantization_kernel_t)

    explicit jit_brgemm_src_quantization_kernel_t(const src_quantization_compile_params_t& jcp);

private:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                                         Xbyak::Xmm,
                                                         isa == dnnl::impl::cpu::x64::avx2,
                                                         Xbyak::Ymm,
                                                         Xbyak::Zmm>::type;

    void generate() override;
    void load_src(Vmm vmm_load, const Xbyak::Address& addr);

    enum class op_type : uint8_t { max, sum };
    void horiz_op(Vmm vmm_src_arg, Vmm vmm_aux_arg, op_type type);

    Vmm vmm_src() {
        return Vmm(0);
    }
    Vmm vmm_max() {
        return Vmm(1);
    }
    Vmm vmm_sign_bit_mask() {
        return Vmm(2);
    }
    Vmm vmm_aux() {
        return Vmm(3);
    }
    Vmm vmm_int8_max() {
        return Vmm(4);
    }
    Vmm vmm_qscale() {
        return Vmm(5);
    }
    Vmm vmm_one() {
        return Vmm(6);
    }
    Vmm vmm_src_sum_accum() {
        return Vmm(7);
    }

    Xbyak::Reg64 reg_src = Xbyak::Reg64(8);
    Xbyak::Reg64 reg_qsrc = Xbyak::Reg64(9);
    Xbyak::Reg64 reg_src_scales = Xbyak::Reg64(10);
    Xbyak::Reg64 reg_ic_size = Xbyak::Reg64(11);
    Xbyak::Reg64 reg_tmp = Xbyak::Reg64(12);
    Xbyak::Reg64 reg_src_grouped_sum = Xbyak::Reg64(13);

    size_t vec_size;
};

}  // namespace ov::intel_cpu
