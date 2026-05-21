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

struct fused_decomp_matmul_compile_params_t {
    size_t oc_block;       // 32 (AVX2) or 64 (AVX512)
    size_t ic_block;       // rd_block aligned to group size

    dnnl::impl::data_type_t src_dt;    // f32, bf16, or s8 (dyn_quant)
    dnnl::impl::data_type_t wei_dt;    // u8, s8, u4, s4, nf4, f4_e2m1, u2
    dnnl::impl::data_type_t dst_dt;    // f32

    bool with_scales = false;
    bool broadcast_scales = false;
    dnnl::impl::data_type_t scales_dt = dnnl::impl::data_type::f32;

    bool with_zero_points = false;
    bool broadcast_zero_points = false;
    dnnl::impl::data_type_t zero_points_dt = dnnl::impl::data_type::u8;

    bool is_dyn_quant = false;
    bool with_src_grouped_sum = false;
};

struct fused_decomp_matmul_runtime_params_t {
    const void* src_ptr;
    const void* wei_ptr;
    void* dst_ptr;
    const void* scales_ptr;
    const void* zero_points_ptr;
    const void* src_scales_ptr;
    const void* src_grouped_sum_ptr;
    size_t ic_size;
    size_t is_accumulate;  // 0=zero dst first, 1=accumulate
};

struct jit_fused_decomp_matmul_kernel_base_t {
    void operator()(const fused_decomp_matmul_runtime_params_t* args) const {
        assert(ker_);
        ker_(args);
    }

    explicit jit_fused_decomp_matmul_kernel_base_t(const fused_decomp_matmul_compile_params_t& jcp) : jcp_(jcp) {}
    virtual ~jit_fused_decomp_matmul_kernel_base_t() = default;

protected:
    void (*ker_)(const fused_decomp_matmul_runtime_params_t*) = nullptr;
    fused_decomp_matmul_compile_params_t jcp_;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
struct jit_fused_decomp_matmul_kernel_t : public jit_fused_decomp_matmul_kernel_base_t,
                                           public dnnl::impl::cpu::x64::jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_fused_decomp_matmul_kernel_t)

    explicit jit_fused_decomp_matmul_kernel_t(const fused_decomp_matmul_compile_params_t& jcp);

private:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                                         Xbyak::Xmm,
                                                         isa == dnnl::impl::cpu::x64::avx2,
                                                         Xbyak::Ymm,
                                                         Xbyak::Zmm>::type;

    static constexpr int n_vregs = dnnl::impl::cpu::x64::cpu_isa_traits_t<isa>::n_vregs;
    static constexpr int vlen = dnnl::impl::cpu::x64::cpu_isa_traits_t<isa>::vlen;
    static constexpr int simd_w = vlen / sizeof(float);

    void generate() override;
    void generate_float_path();
    void generate_dyn_quant_path();

    void load_weights_float(Vmm vmm_load, const Xbyak::Address& addr, int ic_sub);
    void load_weights_u8_for_vnni(Vmm vmm_load, const Xbyak::Address& addr, int rd_offset);

    void init_lookup_tables();
    void load_scales(int num_oc_regs);
    void load_zero_points(int num_oc_regs);

    // Accumulator registers
    Vmm vmm_acc(int ocb) { return Vmm(ocb); }

    // Scale/ZP registers (loaded per IC group, shared across IC iterations)
    int nb_oc_regs() const { return static_cast<int>(jcp_.oc_block / simd_w); }

    Vmm vmm_wei_scale(int ocb) { return Vmm(nb_oc_regs() + ocb); }
    Vmm vmm_wei_zp(int ocb) { return Vmm(2 * nb_oc_regs() + ocb); }

    // Working registers (allocated from top)
    Vmm vmm_src_bcast() { return Vmm(n_vregs - 1); }
    Vmm vmm_wei_load() { return Vmm(n_vregs - 2); }
    Vmm vmm_tmp0() { return Vmm(n_vregs - 3); }
    Vmm vmm_tmp1() { return Vmm(n_vregs - 4); }

    // Lookup table / mask registers
    Vmm vmm_lookup() { return Vmm(n_vregs - 5); }
    Vmm vmm_lookup_low() { return Vmm(n_vregs - 5); }
    Vmm vmm_lookup_high() { return Vmm(n_vregs - 6); }
    Vmm vmm_mask_val() { return Vmm(n_vregs - 7); }
    Vmm vmm_mask8() { return Vmm(n_vregs - 8); }
    Vmm vmm_mask7() { return Vmm(n_vregs - 9); }

    // GPRs
    Xbyak::Reg64 reg_src = Xbyak::Reg64(8);
    Xbyak::Reg64 reg_wei = Xbyak::Reg64(9);
    Xbyak::Reg64 reg_dst = Xbyak::Reg64(10);
    Xbyak::Reg64 reg_scales = Xbyak::Reg64(11);
    Xbyak::Reg64 reg_zero_points = Xbyak::Reg64(12);
    Xbyak::Reg64 reg_ic_size = Xbyak::Reg64(13);
    Xbyak::Reg64 reg_tmp = Xbyak::Reg64(14);
    Xbyak::Reg64 reg_src_scales = Xbyak::Reg64(15);

    size_t get_typesize_scale() const;
    size_t get_wei_element_stride() const;
};

}  // namespace ov::intel_cpu
