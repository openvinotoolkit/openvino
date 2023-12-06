// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_brgemm_copy_b_emitter.hpp"

#include "jit_brgemm_emitter.hpp"

#include "snippets/utils.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/port_connector.hpp"

#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

#include <cpu/x64/brgemm/brgemm.hpp>
#include <cpu/x64/matmul/brgemm_matmul_copy_utils.hpp>
#include <cpu/x64/matmul/brgemm_matmul_utils.hpp>
#include <cpu/x64/amx_tile_configure.hpp>


using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {


jit_brgemm_copy_b_emitter::jit_brgemm_copy_b_emitter(jit_generator* h, cpu_isa_t isa, const  ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto brgemm_repack = ov::as_type_ptr<ov::intel_cpu::BrgemmCopyB>(expr->get_node());
    if (!brgemm_repack)
        OPENVINO_THROW("jit_brgemm_copy_b_emitters expects BrgemmCopyB node");

    m_brgemm_prc_in0 = brgemm_repack->get_src_element_type();
    m_brgemm_prc_in1 = brgemm_repack->get_input_element_type(0);
    m_brgemmVNNIFactor = 4 / m_brgemm_prc_in0.size();
    m_with_comp = brgemm_repack->is_with_compensations();
    m_in_offset = brgemm_repack->get_offset_in();
    m_out_offset = brgemm_repack->get_offset_out();
    if (m_with_comp)
        m_comp_offset = brgemm_repack->get_offset_compensations();

    const auto& in_desc = expr->get_input_port_descriptor(0);
    const auto& layout = in_desc->get_layout();
    const auto& original_shape = in_desc->get_shape();
    auto transposed_shape = original_shape;
    size_t leading_dimension = *(original_shape.rbegin());
    if (!layout.empty()) {
        transposed_shape = snippets::utils::get_planar_vdims(original_shape, layout);
        leading_dimension = jit_brgemm_emitter::get_in_leading_dim(original_shape, layout);
    }

    m_N = *(transposed_shape.rbegin());
    m_K = *(transposed_shape.rbegin() + 1);

    m_N_blk = brgemm_repack->get_n_block_size();
    m_K_blk = brgemm_repack->get_k_block_size();

    m_N_tail = m_N % m_N_blk;
    m_K_tail = m_K % m_K_blk;
    m_LDB = m_brgemm_prc_in1 == ov::element::f32 ? leading_dimension : rnd_up(m_N, m_N_blk);

    const auto dt_in0 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(m_brgemm_prc_in0));
    const auto dt_in1 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(m_brgemm_prc_in1));

    const bool isAMXSupported = mayiuse(avx512_core_amx);
    const auto use_amx = isAMXSupported && m_brgemm_prc_in0 != ov::element::f32 && (m_K % m_brgemmVNNIFactor == 0) && (m_N % m_brgemmVNNIFactor == 0);
    init_brgemm_copy(m_kernel, leading_dimension, m_N_blk, m_N_tail, m_LDB, m_K - m_K_tail, use_amx, dt_in0, dt_in1);
}

void jit_brgemm_copy_b_emitter::init_brgemm_copy(std::unique_ptr<matmul::jit_brgemm_matmul_copy_b_t>& kernel,
                                          size_t N, size_t N_blk, size_t N_tail, size_t LDB, size_t K,
                                          bool is_with_amx, dnnl_data_type_t dt_in0, dnnl_data_type_t dt_in1) const {
    matmul::brgemm_matmul_conf_t brgCopyKernelConf;
    brgCopyKernelConf.src_dt = dt_in0;
    brgCopyKernelConf.wei_dt = dt_in1;
    brgCopyKernelConf.wei_n_blk = static_cast<int>(N_blk);
    brgCopyKernelConf.wei_tag = dnnl_abcd;  // What's about other ranks?
    brgCopyKernelConf.copy_B_wei_stride = 0;
    brgCopyKernelConf.LDB = static_cast<dim_t>(LDB);
    brgCopyKernelConf.N =  static_cast<dim_t>(N);
    brgCopyKernelConf.N_tail =  static_cast<dim_t>(N_tail);
    brgCopyKernelConf.N_blk =  static_cast<dim_t>(N_blk);
    brgCopyKernelConf.K =  static_cast<dim_t>(K);
    brgCopyKernelConf.K_blk =  static_cast<dim_t>(K);
    brgCopyKernelConf.N_chunk_elems = brgCopyKernelConf.N_blk;
    brgCopyKernelConf.b_dt_sz = DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(brgCopyKernelConf.src_dt));
    brgCopyKernelConf.tr_b_dt_sz = DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(brgCopyKernelConf.src_dt));
    brgCopyKernelConf.req_wei_vnni_downconvert = false;

    if (is_with_amx) {
        brgCopyKernelConf.isa = avx512_core_amx;
        brgCopyKernelConf.s8s8_compensation_required = false;
    } else {
        brgCopyKernelConf.isa = dt_in0 == dnnl_data_type_t::dnnl_bf16 ? avx512_core_bf16 : avx512_core_vnni;
        brgCopyKernelConf.s8s8_compensation_required = dt_in0 == dnnl_data_type_t::dnnl_s8;
    }

    brgCopyKernelConf.has_zero_point_a = false;
    brgCopyKernelConf.has_zero_point_b = false;
    brgCopyKernelConf.src_zp_type = dnnl::impl::cpu::x64::none;

    auto status = matmul::create_brgemm_matmul_copy_b(kernel, &brgCopyKernelConf);
    if (status != dnnl_success)
        OPENVINO_THROW("BrgemmRepackEmitter cannot create kernel due to invalid params");
}

void jit_brgemm_copy_b_emitter::emit_impl(const std::vector<size_t>& in,
                                   const std::vector<size_t>& out) const {
    if (host_isa_ == cpu::x64::avx512_core) {
        Xbyak::Reg64 src(static_cast<int>(in[0]));
        Xbyak::Reg64 dst(static_cast<int>(out[0]));
        Xbyak::Reg64 comp(static_cast<int>(0));  // Compensations. Default reg idx is 0 if there aren't the compensations
        if (m_with_comp) {
            if (out.size() != 2) {
                OPENVINO_THROW("jit_brgemm_copy_b_emitter with compensations requires separate register for them");
            }
            comp = Xbyak::Reg64(static_cast<int>(out[1]));
        }

        const size_t data_size = m_brgemm_prc_in1.size();
        for (size_t nb = 0; nb < div_up(m_N, m_N_blk); nb++) {
            const size_t offset_in = m_in_offset + nb * m_N_blk * data_size;
            const size_t offset_out = m_out_offset + nb * m_N_blk * m_brgemmVNNIFactor * data_size;
            const size_t offset_comp = m_with_comp ? m_comp_offset + nb * m_N_blk * sizeof(int32_t) : 0;

            const bool is_N_tail = (m_N - nb * m_N_blk < m_N_blk);
            const auto current_N_blk = is_N_tail ? m_N_tail : m_N_blk;

            emit_kernel_call(m_kernel.get(), src, dst, comp, current_N_blk, m_K, offset_in, offset_out, offset_comp);
        }
    } else {
        OPENVINO_THROW("jit_brgemm_copy_b_emitter requires at least avx512_core instruction set");
    }
}

void jit_brgemm_copy_b_emitter::emit_kernel_call(const matmul::jit_brgemm_matmul_copy_b_t* kernel, Reg64 src, Reg64 dst, Reg64 comp,
                                          size_t N, size_t K, size_t offset_in, size_t offset_out, size_t offset_comp) const {
    constexpr size_t gpr_size = 8;
    Xbyak::Operand gprs_to_save[] = {h->r8, h->r9, h->r10, h->r11, h->r12, h->r13, h->r14, h->r15,
                                     h->rax, h->rcx, h->rdx, h->rdi, h->rsi, h->rbp, h->rbx};
    size_t n_gprs_to_save = sizeof(gprs_to_save) / sizeof(gprs_to_save[0]);

    h->sub(h->rsp, n_gprs_to_save * gpr_size);
    for (size_t i = 0; i < n_gprs_to_save; ++i)
        h->mov(h->ptr[h->rsp + i * gpr_size], gprs_to_save[i]);

    // caller obligation to save k-regs as callee may use them
    size_t n_k_regs_to_save = 8;
    h->sub(h->rsp, n_k_regs_to_save * k_mask_size);
    for (size_t i = 0; i < n_k_regs_to_save; ++i) {
        if (mayiuse(avx512_core))
            h->kmovq(h->ptr[h->rsp + i * k_mask_size], Opmask(static_cast<int>(i)));
        else
            h->kmovw(h->ptr[h->rsp + i * k_mask_size], Opmask(static_cast<int>(i)));
    }

    // 1. Caller obligation to save vector registers as callee may use them.
    // 2. There is an implicit assumption that the host code uses the same
    // `isa` as the injector. Once the assumption is wrong, `vecs_count` and
    // `vlen` should be replaced with `host_isa::vlen` and
    // `host_isa::vecs_count`.
    h->sub(h->rsp, get_max_vecs_count() * get_vec_length());
    for (size_t i = 0; i < get_max_vecs_count(); ++i)
        h->uni_vmovups(h->ptr[h->rsp + i * get_vec_length()], Zmm(i));

    const auto data_ptr = [&](Xmm xmm, Xbyak::Reg64 reg, size_t bytes_offset) {
        h->uni_vmovq(reg, xmm);
        if (bytes_offset) h->add(reg, bytes_offset);
    };
#ifdef _WIN32
    const auto push_value = [&](size_t value, size_t index) {
        // Firstly we need to move integer to GPR. Then we can move value from GPR to stack
        h->mov(abi_not_param1, value);
        h->mov(h->qword[h->rsp + index * gpr_size], abi_not_param1);
    };
#endif

    // save function address in gpr to pass in call instruction
    const auto &kernel_overload = static_cast<void (*)(matmul::jit_brgemm_matmul_copy_b_t*,
                                                       const void*,
                                                       const void*,
                                                       const void*,
                                                       size_t,
                                                       size_t)>(execute);
    h->mov(h->rbp, reinterpret_cast<uintptr_t>(kernel_overload));
    // todo: several of addr_{A, B, C} could be also abi_paramX, so one of them could be corrupted
    //  if moving directly h->uni_vmovq(abi_paramX, adr_X). Save them to vector regs to avoid corruption.
    //  It's likely that a more efficient solution exists.
    h->uni_vmovq(Xmm(0), src);
    h->uni_vmovq(Xmm(1), dst);
    if (m_with_comp)
        h->uni_vmovq(Xmm(2), comp);
    // todo: Windows ABI : requires different num of arguments passed in regs and on the stack. Need to align.
    h->mov(abi_param1, reinterpret_cast<uintptr_t>(kernel));

    data_ptr(Xmm(0), abi_param2, offset_in);
    data_ptr(Xmm(1), abi_param3, offset_out);
    if (m_with_comp) {
        data_ptr(Xmm(2), abi_param4, offset_comp);
    } else {
        h->mov(abi_param4, reinterpret_cast<uintptr_t>(nullptr));
    }

#ifdef _WIN32
    // Before function call we should allocate stack area for
    //  - register parameters - ABI parameters (shadow space)
    //  - stack parameters - remaining parameters
    const size_t num_args_passed_on_stack = 6;  // count of function kernel_overload() parameters
    size_t abi_param_count = sizeof(abi_param_regs) / sizeof(abi_param_regs[0]);

    h->sub(h->rsp, num_args_passed_on_stack * gpr_size);
    push_value(N, abi_param_count + 0);
    push_value(K, abi_param_count + 1);
#else
    h->mov(abi_param5, N);
    h->mov(abi_param6, K);
#endif
    // align stack on 16-byte as ABI requires
    // note that RBX must not be changed by the callee
    h->mov(h->rbx, h->rsp);
    h->and_(h->rbx, 0xf);
    h->sub(h->rsp, h->rbx);

    h->call(h->rbp);

    h->add(h->rsp, h->rbx);

#ifdef _WIN32
        h->add(h->rsp, gpr_size * num_args_passed_on_stack);
#endif
    // restore vector registers
    for (int i = static_cast<int>(get_max_vecs_count()) - 1; i >= 0; --i) {
        h->uni_vmovups(Zmm(i), h->ptr[h->rsp + i * get_vec_length()]);
    }
    h->add(h->rsp, (get_max_vecs_count()) * get_vec_length());

    // restore k registers
    for (int i = n_k_regs_to_save - 1; i >= 0; --i) {
        if (mayiuse(avx512_core))
            h->kmovq(Opmask(i), h->ptr[h->rsp + i * k_mask_size]);
        else
            h->kmovw(Opmask(i), h->ptr[h->rsp + i * k_mask_size]);
    }
    h->add(h->rsp, n_k_regs_to_save * k_mask_size);

    // restore gpr registers
    for (int i = n_gprs_to_save - 1; i >= 0; --i)
        h->mov(gprs_to_save[i], h->ptr[h->rsp + i * gpr_size]);
    h->add(h->rsp, n_gprs_to_save * gpr_size);
}

void jit_brgemm_copy_b_emitter::execute(matmul::jit_brgemm_matmul_copy_b_t *kernel, const void *src,
                                 const void *dst, const void *comp, size_t N, size_t K) {
    if (!kernel)
        OPENVINO_THROW("Kernel for `brgemm_copy_b` hasn't been created");

    auto ctx = dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t::ctx_t();
    ctx.current_N_blk = N;
    ctx.src = src;
    ctx.tr_src = dst;
    ctx.compensation_ptr = comp;
    ctx.zp_a_compensation_ptr = nullptr;
    ctx.zp_a_neg_value_ptr = nullptr;
    ctx.current_K_start = 0;
    ctx.current_K_iters = K;

    (*kernel)(&ctx);
}

}   // namespace intel_cpu
}   // namespace ov
