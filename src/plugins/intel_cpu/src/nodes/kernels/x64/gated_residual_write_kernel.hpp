// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <xbyak/xbyak.h>

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "jit_kernel_base.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::kernel {

// GatedResidual WRITE mode for one (row, branch): out[d] = residual[d] + gate * y[d], d in [0, data_size).
// residual, y, gate and out share one element type; the gate is a single element that the kernel loads and
// broadcasts itself. The arithmetic is f32 with a single rounding on store.
struct jit_gated_residual_write_compile_params {
    ov::element::Type prc;
    size_t data_size = 0UL;
};

struct jit_gated_residual_write_call_args {
    const uint8_t* residual;
    const uint8_t* y;
    const uint8_t* gate;  // one element at the same precision
    uint8_t* out;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
struct jit_gated_residual_write_kernel
    : public JitKernel<jit_gated_residual_write_compile_params, jit_gated_residual_write_call_args> {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_gated_residual_write_kernel)

    static constexpr size_t vec_size = dnnl::impl::cpu::x64::cpu_isa_traits_t<isa>::vlen / sizeof(float);

    explicit jit_gated_residual_write_kernel(const jit_gated_residual_write_compile_params& jcp)
        : JitKernel(jit_name(), jcp, isa) {}

private:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                                         Xbyak::Xmm,
                                                         isa == dnnl::impl::cpu::x64::avx2,
                                                         Xbyak::Ymm,
                                                         Xbyak::Zmm>::type;

    void generate() override;
    // out = residual + gate * y for `step` elements, advancing the pointers when it is a full vector.
    void load_fma_store(size_t step);
    void load(const Vmm& vmm_dst, const Xbyak::Reg64& reg_src, ov::element::Type src_prc, const int& elt_num);
    void store(const Xbyak::Reg64& reg_dst, const Vmm& vmm_src, ov::element::Type dst_prc, const int& elt_num);

    const Vmm vmm_gate = Vmm(0);
    const Vmm vmm_residual = Vmm(1);
    const Vmm vmm_y = Vmm(2);
    const Xbyak::Reg64 reg_residual = r8;
    const Xbyak::Reg64 reg_y = r10;
    const Xbyak::Reg64 reg_gate = r13;
    const Xbyak::Reg64 reg_out = r11;
    const Xbyak::Reg64 reg_work_amount = r12;

    std::unordered_map<size_t, std::unique_ptr<jit_emitter>> emitters;
    const std::vector<size_t> pool_aux_gpr_idxs = {static_cast<size_t>(rax.getIdx()), static_cast<size_t>(r9.getIdx())};
    const std::vector<size_t> pool_aux_vmm_idxs = {7};
};

}  // namespace ov::intel_cpu::kernel
