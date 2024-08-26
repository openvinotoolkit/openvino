// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_brgemm_copy_b_emitter.hpp"

#include "jit_brgemm_emitter.hpp"

#include "snippets/utils/utils.hpp"
#include "snippets/lowered/expression.hpp"

#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

#include <cpu/x64/brgemm/brgemm.hpp>
#include <cpu/x64/matmul/brgemm_matmul_utils.hpp>


using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace ov::intel_cpu::brgemm_utils;
using namespace ov::snippets::utils;

namespace ov {
namespace intel_cpu {
jit_brgemm_copy_b_emitter::jit_brgemm_copy_b_emitter(jit_generator* h, cpu_isa_t isa, const  ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto brgemm_repack = ov::as_type_ptr<ov::intel_cpu::BrgemmCopyB>(expr->get_node());
    if (!brgemm_repack)
        OV_CPU_JIT_EMITTER_THROW("expects BrgemmCopyB node");
    OV_CPU_JIT_EMITTER_ASSERT(is_superset(host_isa_, cpu::x64::avx2), "host_isa must be at least avx2");
    m_with_comp = with_compensations(brgemm_repack->get_type());
    m_in_offset = brgemm_repack->get_offset_in();
    m_out_offset = brgemm_repack->get_offset_out();
    if (m_with_comp)
        m_comp_offset = brgemm_repack->get_offset_compensations();

    const auto& in_desc = expr->get_input_port_descriptor(0);
    const auto& original_shape = in_desc->get_shape();
    const auto& layout = in_desc->get_layout();
    m_transpose = !layout.empty() && layout.back() != layout.size() - 1;
    if (m_transpose)
        OPENVINO_ASSERT(layout[layout.size() - 2] == layout.size() - 1, "supports only N dim placed as last or pre last dimension");

    const auto planar_shape = get_planar_vdims(original_shape, layout);
    const size_t N = *planar_shape.rbegin();
    m_K = *++planar_shape.rbegin();
    OV_CPU_JIT_EMITTER_ASSERT(!is_dynamic_value(N) && !is_dynamic_value(m_K), "K and N dims must be static");

    const auto& in_subtensor = get_projected_subtensor(expr->get_input_port(0));
    m_N_blk = *in_subtensor.rbegin();
    m_K_blk = *++in_subtensor.rbegin();
    OV_CPU_JIT_EMITTER_ASSERT(m_N_blk <= N && m_K_blk <= m_K, "BrgemmCopyB has incompatible subtensor dimensions");
    m_brg_weight_etype = brgemm_repack->get_input_element_type(0);
    m_inner_N_block = repacking::compute_inner_n_block(m_brg_weight_etype);
    m_inner_N_tail = m_N_blk % m_inner_N_block;
    m_brgemmVNNIFactor = compute_vnni_factor(m_brg_weight_etype);

    OV_CPU_JIT_EMITTER_ASSERT(m_brgemmVNNIFactor > 0, "brgemmVNNIFactor value must be positive.");
    OV_CPU_JIT_EMITTER_ASSERT(m_K_blk == m_K || m_K_blk % m_brgemmVNNIFactor == 0,
                              "K Block size (", m_K_blk, "), which is not divisible by brgemmVNNIFactor (",
                              m_brgemmVNNIFactor, ") and not equal to K dimension (", m_K,
                              "), is not supported for brgemm data repacking.");

    OV_CPU_JIT_EMITTER_ASSERT(get_projected_subtensor(expr->get_output_port(0)) == in_subtensor,
                              "output and input subtensors must be equal");
    if (m_with_comp) {
        const auto& compensations_subtensor = get_projected_subtensor(expr->get_output_port(1));
        const auto& compensations_n = *compensations_subtensor.rbegin();
        const auto& compensations_k = *++compensations_subtensor.rbegin();
        OV_CPU_JIT_EMITTER_ASSERT(compensations_n == m_N_blk && compensations_k == 1,
                                  "compensations subtensor must be {1, m_N_blk}");
    }

    const auto& brg_src_etype = brgemm_repack->get_src_element_type();
    OV_CPU_JIT_EMITTER_ASSERT(one_of(m_brg_weight_etype, element::f32, element::bf16, element::i8), "doesn't support precision ", m_brg_weight_etype);

    const auto brgemm_type = get_brgemm_type(brg_src_etype, m_K_blk, m_N_blk, m_transpose);

    const auto ldb = repacking::compute_out_leading_dim(m_N_blk, m_brg_weight_etype);
    const auto wei_stride = get_dim_stride(expr->get_input_port(0), m_transpose ? 0 : 1) * m_brg_weight_etype.size();
    // Note: 2D format tags are used just to force the needed OneDNN primitive creation.
    // However, the generated primitive can be also applied to tensors with other ranks
    const auto format = m_transpose ? dnnl_ba : dnnl_ab;
    init_brgemm_copy(m_kernel, N, m_inner_N_block, m_inner_N_tail, ldb, m_K_blk, brgemm_type, brg_src_etype, m_brg_weight_etype, wei_stride, format);
}

void jit_brgemm_copy_b_emitter::init_brgemm_copy(std::unique_ptr<matmul::jit_brgemm_matmul_copy_b_t>& kernel,
                                                 size_t N, size_t N_blk, size_t N_tail, size_t out_leading_dim, size_t K_blk, BRGEMM_TYPE brgemm_type,
                                                 const ov::element::Type& src_dt, const ov::element::Type& wei_dt, size_t wei_stride,
                                                 dnnl_format_tag_t format) const {
    matmul::brgemm_matmul_conf_t brgCopyKernelConf;
    brgCopyKernelConf.src_dt = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(src_dt));
    brgCopyKernelConf.wei_dt = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(wei_dt));
    brgCopyKernelConf.orig_wei_dt = brgCopyKernelConf.wei_dt;
    brgCopyKernelConf.wei_n_blk = static_cast<int>(N_blk);
    brgCopyKernelConf.wei_tag = format;
    brgCopyKernelConf.transposed_B = m_transpose;
    brgCopyKernelConf.copy_B_wei_stride = wei_stride;
    brgCopyKernelConf.LDB = static_cast<dim_t>(out_leading_dim);
    brgCopyKernelConf.N =  static_cast<dim_t>(N);
    brgCopyKernelConf.N_tail = static_cast<dim_t>(N_tail);
    brgCopyKernelConf.N_blk =  static_cast<dim_t>(N_blk);
    brgCopyKernelConf.K =  static_cast<dim_t>(K_blk);
    brgCopyKernelConf.K_blk =  static_cast<dim_t>(K_blk);
    brgCopyKernelConf.N_chunk_elems = brgCopyKernelConf.N_blk;
    brgCopyKernelConf.b_dt_sz = DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(brgCopyKernelConf.src_dt));
    brgCopyKernelConf.tr_b_dt_sz = DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(brgCopyKernelConf.src_dt));

    brgCopyKernelConf.req_wei_vnni_downconvert = false;


    brgCopyKernelConf.isa = get_primitive_isa(src_dt, with_amx(brgemm_type));
    brgCopyKernelConf.s8s8_compensation_required = with_compensations(brgemm_type);

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

    Xbyak::Reg64 src(static_cast<int>(in[0]));
    Xbyak::Reg64 dst(static_cast<int>(out[0]));
    Xbyak::Reg64 comp(static_cast<int>(m_with_comp ? out[1] : 0));

    const size_t data_size = m_brg_weight_etype.size();
    size_t start_in = m_in_offset;
    size_t start_out = m_out_offset;
    size_t start_comp = m_comp_offset;

    // OneDNN requires tail handling before main iterations
    if (m_inner_N_tail != 0) {
        emit_kernel_call(m_kernel.get(), src, dst, comp, m_inner_N_tail, m_K_blk, start_in, start_out, start_comp);
        start_in += m_transpose ? m_K * m_inner_N_tail * data_size : m_inner_N_tail * data_size;
        start_out += m_inner_N_tail * m_brgemmVNNIFactor * data_size;
        start_comp += m_inner_N_tail * sizeof(int32_t);
    }

    const size_t in_ld = m_transpose ? m_K * m_inner_N_block * data_size : m_inner_N_block * data_size;
    const size_t out_ld = m_inner_N_block * m_brgemmVNNIFactor * data_size;
    const size_t comp_ld = m_inner_N_block * sizeof(int32_t);
    for (size_t nb = 0; nb < m_N_blk / m_inner_N_block; nb++) {
        const size_t offset_in = start_in + nb * in_ld;
        const size_t offset_out = start_out + nb * out_ld;
        const size_t offset_comp = m_with_comp ? start_comp + nb * comp_ld : 0;
        emit_kernel_call(m_kernel.get(), src, dst, comp, m_inner_N_block, m_K_blk, offset_in, offset_out, offset_comp);
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
