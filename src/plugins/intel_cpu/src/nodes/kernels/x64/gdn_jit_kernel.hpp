// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <xbyak/xbyak.h>

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "jit_kernel_base.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::kernel {

struct jit_gdn_compile_params {
    ov::element::Type data_prc = ov::element::f32;
    size_t qk_head_size = 0;
    size_t v_tile = 1;
    bool fuse_qk_l2norm = false;
    float q_l2_norm_eps = 1e-6F;
    float k_l2_norm_eps = 1e-6F;
    float q_scale = 1.0F;
};

struct jit_gdn_call_args {
    uint8_t* state;
    const uint8_t* key_seq;
    const uint8_t* query_seq;
    const uint8_t* value_seq;
    const uint8_t* gate_seq;
    const uint8_t* beta_seq;
    size_t t_size;
    size_t key_query_stride;
    size_t gate_beta_stride;
    size_t value_stride;
    size_t output_stride;
    uint8_t* key_tmp;
    uint8_t* query_tmp;
    uint8_t* output_seq;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
struct jit_gdn_kernel : public JitKernel<jit_gdn_compile_params, jit_gdn_call_args> {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_gdn_kernel)

    explicit jit_gdn_kernel(const jit_gdn_compile_params& jcp) : JitKernel(jit_name(), jcp, isa) {}

private:
    using Xmm = Xbyak::Xmm;
    using Vmm = std::conditional_t<isa == dnnl::impl::cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>;

    static constexpr size_t vec_size = dnnl::impl::cpu::x64::cpu_isa_traits_t<isa>::vlen / sizeof(float);
    static constexpr size_t vec_bytes = vec_size * sizeof(float);
    static constexpr int vec_shift = isa == dnnl::impl::cpu::x64::avx2 ? 3 : 4;

    // GPR map
    const Xbyak::Reg64 reg_args = rbx;
    const Xbyak::Reg64 reg_state = r8;
    const Xbyak::Reg64 reg_key_tmp = r9;
    const Xbyak::Reg64 reg_query_tmp = r10;
    const Xbyak::Reg64 reg_t = r12;
    const Xbyak::Reg64 reg_key_seq = r13;
    const Xbyak::Reg64 reg_query_seq = r14;
    const Xbyak::Reg64 reg_value_seq = r15;
    const Xbyak::Reg64 reg_aux = r11;
    const Xbyak::Reg64 reg_one = rdx;
    const Xbyak::Reg64 reg_gate_seq = rsi;
    const Xbyak::Reg64 reg_beta_seq = rdi;
    const Xbyak::Reg64 reg_out_seq = rbp;
    const Xbyak::Reg64 reg_aux2 = rax;

    // XMM map
    const Xmm x_hk = Xmm(0);
    const Xmm x_tmp0 = Xmm(1);
    const Xmm x_tmp1 = Xmm(2);
    const Xmm x_delta = Xmm(3);
    const Xmm x_out = Xmm(4);
    const Xmm x_gate = Xmm(5);
    const Xmm x_beta = Xmm(6);
    const Xmm x_value = Xmm(7);
    const Xmm x_eps_k = Xmm(8);
    const Xmm x_eps_q = Xmm(9);
    const Xmm x_qscale = Xmm(10);

    const Vmm v_tmp0 = Vmm(x_tmp0.getIdx());
    const Vmm v_tmp1 = Vmm(x_tmp1.getIdx());
    const Vmm v_aux0 = Vmm(11);
    const Vmm v_aux1 = Vmm(12);
    const Vmm v_aux2 = Vmm(13);

    // Register-based Q/K/H storage for native f16
    // Supports head_dims that are multiples of 32, up to 128
    static constexpr int XF16_ELEMS_PER_ZMM = 32;  // 32 xf16 elements per ZMM register
    static constexpr int MAX_REGS_PER_VEC = 4;     // Max ZMMs per vector (for head_dims=128)

    const Vmm v_q[MAX_REGS_PER_VEC] = {Vmm(14), Vmm(15), Vmm(16), Vmm(17)};  // Query
    const Vmm v_k[MAX_REGS_PER_VEC] = {Vmm(18), Vmm(19), Vmm(20), Vmm(21)};  // Key
    const Vmm v_h[MAX_REGS_PER_VEC] = {Vmm(22), Vmm(23), Vmm(24), Vmm(25)};  // Hidden state

    void generate() override;

    // Native xf16 helpers - f16/bf16, head_dims must be multiple of 32
    void load_vector_native_xf16(Vmm* vmm_array, const Xbyak::Reg64& reg_src, int num_regs);
    void store_vector_native_xf16(const Xbyak::Reg64& reg_dst, Vmm* vmm_array, int num_regs);
    void dot_product_native_xf16(const Xbyak::Xmm& xmm_dst, Vmm* vmm_a, Vmm* vmm_b, int num_regs);
    void scale_vector_native_xf16(Vmm* vmm_array, const Xbyak::Xmm& xmm_scalar, int num_regs);
    void fmadd_vector_native_xf16(Vmm* vmm_dst, Vmm* vmm_src, const Xbyak::Xmm& xmm_scalar, int num_regs);
    void l2norm_inplace_native_xf16(Vmm* vmm_array, const Xbyak::Xmm& xmm_eps, int num_regs);

    // Buffer-based helpers for qk_head_size > 128
    void l2norm_buffer_compute_scale_native_xf16(const Xbyak::Reg64& reg_buffer,
                                                 const Xbyak::Xmm& xmm_eps,
                                                 const Xbyak::Xmm& xmm_scale_out,
                                                 int num_regs,
                                                 int num_chunks);
    void scale_buffer_native_xf16(const Xbyak::Reg64& reg_buffer,
                                  const Xbyak::Xmm& xmm_scale,
                                  Vmm* vmm_temp,
                                  int num_regs,
                                  int num_chunks);
    void load_qk(bool is_f32, bool use_registers, int num_regs, int num_chunks);

    void reduce_zmm_f32_to_xmm_scalar(const Xbyak::Zmm& zmm_src,
                                      const Xbyak::Xmm& xmm_dst,
                                      const Xbyak::Xmm& xmm_tmp0,
                                      const Xbyak::Xmm& xmm_tmp1);
    void dot_product_scalar(const Xbyak::Xmm& xmm_dst,
                            const Xbyak::Reg64& reg_a,
                            const Xbyak::Reg64& reg_b,
                            size_t tail_count,
                            size_t base_off,
                            size_t elem_size,
                            const Xbyak::Xmm& xmm_tmp0,
                            const Xbyak::Xmm& xmm_tmp1);
    void dot_product_to_scalar(const Xbyak::Xmm& xmm_dst, const Xbyak::Reg64& reg_a, const Xbyak::Reg64& reg_b);
    void multiply_scalar(const Xbyak::Reg64& reg_vec, const Xbyak::Xmm& xmm_scalar);
    void l2norm_inplace(const Xbyak::Reg64& reg_vec,
                        const Xbyak::Xmm& xmm_eps,
                        const Xbyak::Xmm& xmm_tmp0,
                        const Xbyak::Xmm& xmm_tmp1,
                        const Xbyak::Xmm& xmm_sum);
    void store(const Xbyak::Reg64& reg_dst,
               const Vmm& vmm_src,
               ov::element::Type dst_prc,
               const int& elt_num,
               size_t offset = 0,
               ov::element::Type src_prc = ov::element::f32);
    void load(const Vmm& vmm_dst,
              const Xbyak::Reg64& reg_src,
              ov::element::Type src_prc,
              const int& elt_num,
              bool fill,
              size_t offset = 0,
              ov::element::Type dst_prc = ov::element::f32);

    std::unordered_map<size_t, std::unique_ptr<jit_emitter>> emitters;
    const std::vector<size_t> pool_aux_gpr_idxs;
    const std::vector<size_t> pool_aux_vmm_idxs;
};

std::shared_ptr<JitKernelBase> create_gdn_jit_kernel(ov::element::Type data_prc = ov::element::f32,
                                                     size_t qk_head_size = 0,
                                                     size_t v_tile = 1,
                                                     bool fuse_qk_l2norm = false,
                                                     float q_l2_norm_eps = 1e-6F,
                                                     float k_l2_norm_eps = 1e-6F);

}  // namespace ov::intel_cpu::kernel
