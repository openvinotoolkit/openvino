// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_kernel_base.hpp"
#include <dnnl_types.h>
#include <cpu_types.h>
#include <ie_precision.hpp>
#include "emitters/jit_load_store_emitters.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

enum ROIAlignLayoutType {
    ncsp,
    blk,
    nspc
};

struct jit_roi_align_params {
    Algorithm alg;
    InferenceEngine::Precision data_prc;
    int data_size;
    ROIAlignLayoutType layout;
    int pooled_h;
    int pooled_w;
};

struct jit_roi_align_call_args {
    // point to srcData for planar
    // point to srcData address list for other layouts
    const void *src;
    const float *weights;
    const float *scale;
    void *buffer;
    void *dst;
    size_t num_samples;
    size_t work_amount;
    size_t src_stride;
};

struct jit_uni_roi_align_kernel {
    void (*ker_)(const jit_roi_align_call_args *);

    void operator()(const jit_roi_align_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_roi_align_kernel(jit_roi_align_params jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_roi_align_kernel() {}

    virtual void create_ker() = 0;

    jit_roi_align_params jcp_;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
struct jit_uni_roi_align_kernel_f32 : public jit_uni_roi_align_kernel, public dnnl::impl::cpu::x64::jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_roi_align_kernel_f32);

    explicit jit_uni_roi_align_kernel_f32(jit_roi_align_params jcp);

    void create_ker() override;

    void generate() override;

private:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41, Xbyak::Xmm, isa == dnnl::impl::cpu::x64::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;

    const int v_len = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen;
    const int x_len = dnnl::impl::cpu::x64::cpu_isa_traits<sse41>::vlen;
    const int v_step = v_len / sizeof(float);
    const int x_step = x_len / sizeof(float);

    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using reg16_t = const Xbyak::Reg16;

    reg64_t reg_src_address = r8;
    // reg_srcx is used after abi parse finised
    reg64_t reg_src0        = r11;
    reg64_t reg_src1        = r12;
    reg64_t reg_src2        = rcx;
    reg64_t reg_src3        = rdi;
    reg64_t reg_weights     = r13;

    reg64_t reg_buf  = r9;
    reg64_t reg_src_stride  = r15;

    reg64_t reg_work_amount = r14;
    reg64_t reg_num_samples = r10;

    reg64_t reg_load_table  = rax;
    reg64_t reg_load_store_mask = rbx;

    reg64_t reg_tmp_64 = rbp;
    reg32_t reg_tmp_32 = ebp;
    reg16_t reg_tmp_16 = bp;

    // [0] for reg_buf
    // [1] for reg_dst
    Xbyak::Xmm xmm_args_pool = Xbyak::Xmm(15);

    std::unordered_map<size_t, std::unique_ptr<jit_emitter>> emitters;

    std::vector<size_t> load_pool_gpr_idxs;
    std::vector<size_t> store_pool_gpr_idxs;
    std::vector<size_t> store_pool_vec_idxs;

    Vmm vmm_zero = Vmm(0);

    // assign for cgather
    Vmm vmm_src0 = Vmm(1);
    Vmm vmm_src1 = Vmm(2);
    Vmm vmm_src2 = Vmm(3);
    Vmm vmm_src3 = Vmm(4);

    Vmm vmm_weights0 = Vmm(5);
    Vmm vmm_weights1 = Vmm(6);
    Vmm vmm_weights2 = Vmm(7);
    Vmm vmm_weights3 = Vmm(8);

    Vmm vmm_sample = Vmm(9);
    Vmm vmm_buf = Vmm(10);
    Vmm vmm_scale = Vmm(11);

    // assign for planar
    reg64_t reg_src = reg_src0;
    reg64_t reg_dst = reg_src1;

    Vmm vmm_weights = vmm_weights0;
    Xbyak::Xmm xmm_weights = Xbyak::Xmm(vmm_weights.getIdx());
    Vmm vmm_src = vmm_src0;
    Xbyak::Xmm xmm_src = Xbyak::Xmm(vmm_src.getIdx());
    Xbyak::Xmm xmm_buf = Xbyak::Xmm(vmm_buf.getIdx());

    Vmm vmm_dst = vmm_src1;
    Xbyak::Xmm xmm_dst = Xbyak::Xmm(vmm_dst.getIdx());
    Vmm vmm_dst_tail = vmm_src2;
    Xbyak::Xmm xmm_dst_tail = Xbyak::Xmm(vmm_dst_tail.getIdx());

    Vmm vmm_temp1 = vmm_weights1;
    Xbyak::Xmm xmm_temp1 = Xbyak::Xmm(vmm_temp1.getIdx());
    Vmm vmm_temp2 = vmm_weights2;
    Xbyak::Xmm xmm_temp2 = Xbyak::Xmm(vmm_temp2.getIdx());

    Vmm vmm_mask = vmm_weights3;

    Xbyak::Opmask k_mask = Xbyak::Opmask(7);

    reg64_t reg_params = abi_param1;

    void emit_emitters_data();
    void load(Xbyak::Reg64 reg_src, Vmm vmm_src, const int elt_num, const int offset = 0);
    void load_buffer(Xbyak::Reg64 reg_src, Vmm vmm_src, const int elt_num, const int offset = 0);
    void load_idx(Xbyak::Reg64 reg_src, Vmm vmm_src, const int elt_num, const int offset = 0);
    void store(Vmm vmm_dst, Xbyak::Reg64 reg_dst, const int elt_num, const int offset = 0);
    void store_buffer(Vmm vmm_dst, Xbyak::Reg64 reg_dst, const int elt_num, const int offset = 0);
    void emit_load(Xbyak::Reg64 reg_src, Vmm vmm_src, Precision src_prc, Precision dst_prc, const int elt_num, const int offset = 0);
    void emit_store(Vmm vmm_dst, Xbyak::Reg64 reg_dst, Precision src_prc, Precision dst_prc, const int elt_num, const int offset = 0);

    void roi_align_cgather();
    void get_src();
    void get_weights();
    void generate_samples(int num);
    void roi_align_planar();

    // gather f32 data from reg_src with vmm_idx(data_size) to vmm_src with f32 precision
    void gather_f32(Vmm &vmm_src, const reg64_t &reg_src, const Vmm &vmm_idx);
    void gather_f32_xmm(Xbyak::Xmm xmm_src, const reg64_t reg_src, const Xbyak::Xmm xmm_idx);

    // gather bf16 data from reg_src with vmm_idx(data_size) to vmm_src with f32 precision
    // bf16 is needed from avx512_core
    void gather_bf16_to_f32_zmm(Vmm vmm_src, const reg64_t reg_src, const Vmm vmm_idx);
    void gather_bf16_to_f32_xmm(Xbyak::Xmm xmm_src, const reg64_t reg_src, const Xbyak::Xmm xmm_idx);
    void horizontal_add_xmm(const Xbyak::Xmm &xmm_dst, const Xbyak::Xmm &xmm_aux);
    void horizontal_add();
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
