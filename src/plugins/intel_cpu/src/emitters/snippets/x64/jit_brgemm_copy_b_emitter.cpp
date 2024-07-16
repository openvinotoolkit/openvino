// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_brgemm_copy_b_emitter.hpp"

#include "jit_brgemm_emitter.hpp"

#include "snippets/utils.hpp"
#include "snippets/lowered/expression.hpp"

#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

#include <cpu/x64/brgemm/brgemm.hpp>
#include <cpu/x64/matmul/brgemm_matmul_utils.hpp>


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
        OV_CPU_JIT_EMITTER_THROW("expects BrgemmCopyB node");

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
        leading_dimension = ov::snippets::utils::get_dim_stride(expr->get_input_port(0));
    }

    const auto& in_subtensor = in_desc->get_subtensor();
    m_N_blk = *in_subtensor.rbegin();
    m_K_blk = *++in_subtensor.rbegin();
    OV_CPU_JIT_EMITTER_ASSERT(m_N_blk <= *transposed_shape.rbegin() && m_K_blk <= *++transposed_shape.rbegin(),
                              "BrgemmCopyB has incompatible subtensor dimensions");
    m_inner_N_block = brgemm_repack->get_n_inner_block_size();
    m_inner_N_tail = m_N_blk % m_inner_N_block;

    OV_CPU_JIT_EMITTER_ASSERT(expr->get_output_port_descriptor(0)->get_subtensor() == in_subtensor, "output and input subtensors must be equal");
    if (m_with_comp) {
        const auto& compensations_subtensor = expr->get_output_port_descriptor(1)->get_subtensor();
        OV_CPU_JIT_EMITTER_ASSERT(
            *compensations_subtensor.rbegin() == m_N_blk && *++compensations_subtensor.rbegin() == 1,
            "compensations subtensor must be {1, m_N_blk}");
    }

    OV_CPU_JIT_EMITTER_ASSERT(!one_of(m_brg_weight_etype, element::bf16, element::i8), "doesn't support precision ", m_brg_weight_etype);
    const auto repacking_buffer_shape = brgemm_repack->get_repacking_buffer_shape();
    OV_CPU_JIT_EMITTER_ASSERT(!repacking_buffer_shape.empty(), "Repacking buffer shape mustn't be empty");
    const auto& LDB = repacking_buffer_shape.back();

    const auto& brg_src_etype = brgemm_repack->get_src_element_type();
    m_brg_weight_etype = brgemm_repack->get_input_element_type(0);
    m_brgemmVNNIFactor = brgemm_repack->get_brgemm_vnni_factor();

    const auto use_amx = mayiuse(avx512_core_amx) && brg_src_etype != ov::element::f32 &&
                         (m_K_blk % m_brgemmVNNIFactor == 0) && (m_N_blk % m_brgemmVNNIFactor == 0);

    const auto src_dt = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(brg_src_etype));
    const auto wei_dt = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(m_brg_weight_etype));

    init_brgemm_copy(m_kernel, leading_dimension, m_inner_N_block, m_inner_N_tail, LDB, m_K_blk, use_amx, src_dt, wei_dt);
}

void jit_brgemm_copy_b_emitter::init_brgemm_copy(std::unique_ptr<matmul::jit_brgemm_matmul_copy_b_t>& kernel,
                                                 size_t N, size_t N_blk, size_t N_tail, size_t LDB, size_t K,
                                                 bool is_with_amx, dnnl_data_type_t src_dt, dnnl_data_type_t wei_dt) const {
    matmul::brgemm_matmul_conf_t brgCopyKernelConf;
    brgCopyKernelConf.src_dt = src_dt;
    brgCopyKernelConf.wei_dt = wei_dt;
    brgCopyKernelConf.wei_n_blk = static_cast<int>(N_blk);
    brgCopyKernelConf.wei_tag = dnnl_abcd;  // What's about other ranks?
    brgCopyKernelConf.copy_B_wei_stride = 0;
    brgCopyKernelConf.LDB = static_cast<dim_t>(LDB);
    brgCopyKernelConf.N =  static_cast<dim_t>(N);
    brgCopyKernelConf.N_tail = static_cast<dim_t>(N_tail);
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
        brgCopyKernelConf.isa = src_dt == dnnl_data_type_t::dnnl_bf16 ? avx512_core_bf16 : avx512_core_vnni;
        brgCopyKernelConf.s8s8_compensation_required = src_dt == dnnl_data_type_t::dnnl_s8;
    }

    brgCopyKernelConf.has_zero_point_a = false;
    brgCopyKernelConf.has_zero_point_b = false;
    brgCopyKernelConf.src_zp_type = dnnl::impl::cpu::x64::none;

    auto status = matmul::create_brgemm_matmul_copy_b(kernel, &brgCopyKernelConf);
    OV_CPU_JIT_EMITTER_ASSERT(status == dnnl_success, "cannot create kernel due to invalid params");
}

void jit_brgemm_copy_b_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == 1, "expects 1 input");
    OV_CPU_JIT_EMITTER_ASSERT((m_with_comp && out.size() == 2) || (!m_with_comp && out.size() == 1),
                              "expects 2 outputs if there are compensations");
}

void jit_brgemm_copy_b_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    validate_arguments(in, out);
    OV_CPU_JIT_EMITTER_ASSERT(host_isa_ == cpu::x64::avx512_core, "requires at least avx512_core instruction set");

    Xbyak::Reg64 src(static_cast<int>(in[0]));
    Xbyak::Reg64 dst(static_cast<int>(out[0]));
    Xbyak::Reg64 comp(static_cast<int>(m_with_comp ? out[1] : 0));

    const size_t data_size = m_brg_weight_etype.size();
    for (size_t nb = 0; nb < div_up(m_N_blk, m_inner_N_block); nb++) {
        const size_t offset_in = m_in_offset + nb * m_inner_N_block * data_size;
        const size_t offset_out = m_out_offset + nb * m_inner_N_block * m_brgemmVNNIFactor * data_size;
        const size_t offset_comp = m_with_comp ? m_comp_offset + nb * m_inner_N_block * sizeof(int32_t) : 0;

        const bool is_N_tail = (m_N_blk - nb * m_inner_N_block < m_inner_N_block);
        const auto current_N_blk = is_N_tail ? m_inner_N_tail : m_inner_N_block;

        emit_kernel_call(m_kernel.get(), src, dst, comp, current_N_blk, m_K_blk, offset_in, offset_out, offset_comp);
    }
}

void jit_brgemm_copy_b_emitter::emit_kernel_call(const matmul::jit_brgemm_matmul_copy_b_t* kernel, Reg64 src, Reg64 dst, Reg64 comp,
                                                 size_t N, size_t K, size_t offset_in, size_t offset_out, size_t offset_comp) const {
    const auto data_ptr = [&](Xmm xmm, Xbyak::Reg64 reg, size_t bytes_offset) {
        h->uni_vmovq(reg, xmm);
        if (bytes_offset) h->add(reg, bytes_offset);
    };

    internal_call_preamble();
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
    // Note: ABI requires that the remaining parameters (except the first for) are pushed to the stack in right-to-left order
    //  Shadow space will be allocated inside internal_call_rsp_align()
    h->push(K);
    h->push(N);
#else
    h->mov(abi_param5, N);
    h->mov(abi_param6, K);
#endif

    internal_call_rsp_align();
    h->call(h->rbp);
    internal_call_rsp_restore();

#ifdef _WIN32
        h->add(h->rsp, gpr_size * 2);
#endif
    internal_call_postamble();
}

void jit_brgemm_copy_b_emitter::execute(matmul::jit_brgemm_matmul_copy_b_t* kernel,
                                        const void* src,
                                        const void* dst,
                                        const void* comp,
                                        size_t N,
                                        size_t K) {
    auto ctx = dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t::ctx_t();
    ctx.current_N_blk = N;
    ctx.src = src;
    ctx.tr_src = dst;
    ctx.compensation_ptr = comp;
    ctx.zp_a_compensation_ptr = nullptr;
    ctx.zp_a_neg_value_ptr = nullptr;
    ctx.current_K_start = 0;
    ctx.current_K_iters = K;

    OV_CPU_JIT_EMITTER_ASSERT(kernel, "Kernel hasn't been created");
    (*kernel)(&ctx);
}

}   // namespace intel_cpu
}   // namespace ov
