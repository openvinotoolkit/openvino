// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_brgemm_emitter.hpp"

#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include <cpu/x64/brgemm/brgemm.hpp>
#include <cpu/x64/amx_tile_configure.hpp>
#include "snippets/utils/utils.hpp"
#include "emitters/plugin/x64/utils.hpp"
#include "utils.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {

jit_brgemm_emitter::jit_brgemm_emitter(jit_generator* h, cpu_isa_t isa,
                                       const ov::snippets::lowered::ExpressionPtr& expr,
                                       const snippets::KernelExecutorTablePtr& kernel_table,
                                       const ov::intel_cpu::MultiCacheWeakPtr& compiled_kernel_cache) :
                                       jit_emitter(h, isa) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto& brgemm_node = as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node());
    const auto& brg0Prc = brgemm_node->get_input_element_type(0);
    const auto& brg1Prc = brgemm_node->get_input_element_type(1);
    const auto brgemm_type = brgemm_node->get_type();
    BrgemmKernelConfig kernel_config(brg0Prc, brg1Prc, with_amx(brgemm_type), with_compensations(brgemm_type),
                                     brgemm_utils::get_primitive_isa(brg0Prc, with_amx(brgemm_type)));
    m_kernel_executor = kernel_table->register_kernel<BrgemmKernelExecutor>(expr,
                                                                            compiled_kernel_cache,
                                                                            kernel_config);
    // Note: even if the Brgemm node is dynamic, the first shapeInfer and RuntimeConfigurator::update()
    // are performed before the BrgemmKernelExecutor registration. So we have to trigger update() manually
    // for both static and the 1st dynamic shapes.
    OV_CPU_JIT_EMITTER_ASSERT(!snippets::utils::is_dynamic_vdims(expr->get_input_port_descriptor(0)->get_shape()) &&
                              !snippets::utils::is_dynamic_vdims(expr->get_input_port_descriptor(1)->get_shape()),
                              "Jit emitter is called when the shapes are unknown");

    m_memory_offsets = {brgemm_node->get_offset_a(), brgemm_node->get_offset_b(), brgemm_node->get_offset_c()};
    m_buffer_ids = {utils::get_buffer_cluster_id(expr->get_input_port(0), m_memory_offsets[0]),
                    utils::get_buffer_cluster_id(expr->get_input_port(1), m_memory_offsets[1]),
                    utils::get_buffer_cluster_id(expr->get_output_port(0), m_memory_offsets[2])};
    if (with_scratchpad(brgemm_type)) {
        m_memory_offsets.push_back(brgemm_node->get_offset_scratch());
        m_buffer_ids.push_back(utils::get_buffer_cluster_id(expr->get_input_port(2), m_memory_offsets.back()));
    }
}

std::set<std::vector<element::Type>> jit_brgemm_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    const auto brgemm = as_type_ptr<ov::intel_cpu::BrgemmCPU>(node);
    OV_CPU_JIT_EMITTER_ASSERT(brgemm, "get_supported_precisions() expects BrgemmCPU node");
    using brgemm_utils::BRGEMM_TYPE;
    if (brgemm->get_type() == BRGEMM_TYPE::STAND_ALONE) {
        return {{element::f32, element::f32}};
    } else if (brgemm->get_type() == BRGEMM_TYPE::REPACKING_ONLY) {
        std::set<std::vector<element::Type>> supported_types = {{element::u8, element::i8},
                                                                {element::bf16, element::bf16},
                                                                {element::f32, element::f32}};
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2_vnni_2))
            supported_types.insert({element::i8, element::i8});
        return supported_types;
    } else if (brgemm->get_type() == BRGEMM_TYPE::WITH_COMPENSATIONS) {
        return {{element::i8, element::i8, element::f32}};
    } else if (brgemm->get_type() == BRGEMM_TYPE::WITH_AMX) {
        return {{element::i8, element::i8, element::u8},
                {element::u8, element::i8, element::u8},
                {element::bf16, element::bf16, element::u8}};
    }
    OV_CPU_JIT_EMITTER_THROW("got BrgemmCPU node with unsupported type");
}

void jit_brgemm_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OV_CPU_JIT_EMITTER_ASSERT(m_memory_offsets.size() == in.size() + 1 && (out.size() == 1),
                              "expects 3 inputs if there are compensations/wsp");
}

void jit_brgemm_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    validate_arguments(in, out);
    std::vector<size_t> mem_ptrs_idxs{in[0], in[1], out[0]};
    if (in.size() > 2)
        mem_ptrs_idxs.emplace_back(in[2]);
    emit_brgemm_kernel_call(mem_ptrs_idxs, m_memory_offsets);
}

void jit_brgemm_emitter::internal_call_preamble() const {
    // gprs
    Xbyak::Operand gprs_to_save[] = {h->r8, h->r9, h->r10, h->r11, h->r12, h->r13, h->r14, h->r15,
                                        h->rax, h->rbx, h->rcx, h->rdx, h->rdi, h->rsi, h->rbp};
    size_t n_gprs_to_save = sizeof(gprs_to_save) / sizeof(gprs_to_save[0]);

    h->sub(h->rsp, n_gprs_to_save * gpr_size);
    for (size_t i = 0; i < n_gprs_to_save; ++i)
        h->mov(h->ptr[h->rsp + i * gpr_size], gprs_to_save[i]);

    // mask regs
    // need preserve based on cpu capability, instead of host isa.
    // in case there are possibilty that different isa emitters exist in one subgraph KernelEmitter from perf standpoint in the future.
    // e.g. other emitters isa is avx512, while this emitter isa is avx2, and internal call is used. Internal call may use avx512 and spoil k-reg.
    // do not care about platform w/ avx512_common but w/o avx512_core(knight landing), which is obsoleted.
    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
        h->sub(h->rsp, k_mask_num * k_mask_size);
        for (size_t i = 0; i < k_mask_num; ++i) {
            h->kmovq(h->ptr[h->rsp + i * k_mask_size], Xbyak::Opmask(static_cast<int>(i)));
        }
    }

    // vector regs
    // 1. Caller obligation to save vector registers as callee may use them.
    // 2. There is an implicit assumption that the host code uses the same
    // `isa` as the injector. Once the assumption is wrong, `vecs_count` and
    // `vlen` should be replaced with `host_isa::vlen` and
    // `host_isa::vecs_count`.
    h->sub(h->rsp, get_max_vecs_count() * get_vec_length());
    for (size_t i = 0; i < get_max_vecs_count(); ++i) {
        push_vec(h->ptr[h->rsp + i * get_vec_length()], i);
    }
}

void jit_brgemm_emitter::internal_call_postamble() const {
    // restore vector registers
    for (int i = static_cast<int>(get_max_vecs_count()) - 1; i >= 0; --i) {
        pop_vec(static_cast<size_t>(i), h->ptr[h->rsp + i * get_vec_length()]);
    }
    h->add(h->rsp, (get_max_vecs_count()) * get_vec_length());

    // restore k reg
    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
        for (int i = k_mask_num - 1; i >= 0; --i) {
            h->kmovq(Xbyak::Opmask(i), h->ptr[h->rsp + i * k_mask_size]);
        }
        h->add(h->rsp, k_mask_num * k_mask_size);
    }

    // restore gpr registers
    Xbyak::Operand gprs_to_save[] = {h->r8, h->r9, h->r10, h->r11, h->r12, h->r13, h->r14, h->r15,
                                        h->rax, h->rbx, h->rcx, h->rdx, h->rdi, h->rsi, h->rbp};
    size_t n_gprs_to_save = sizeof(gprs_to_save) / sizeof(gprs_to_save[0]);
    for (int i = n_gprs_to_save - 1; i >= 0; --i)
        h->mov(gprs_to_save[i], h->ptr[h->rsp + i * gpr_size]);
    h->add(h->rsp, n_gprs_to_save * gpr_size);
}

void jit_brgemm_emitter::internal_call_rsp_align() const {
    h->mov(h->rbx, h->rsp);
    h->and_(h->rbx, 0xf);
    h->sub(h->rsp, h->rbx);
#ifdef _WIN32
    // Allocate shadow space (home space) according to ABI
    h->sub(h->rsp, 32);
#endif
}

void jit_brgemm_emitter::internal_call_rsp_restore() const {
#ifdef _WIN32
    // Release shadow space (home space)
    h->add(h->rsp, 32);
#endif
    h->add(h->rsp, h->rbx);
}

void jit_brgemm_emitter::emit_brgemm_kernel_call(const std::vector<size_t>& mem_ptrs_idxs, const std::vector<size_t>& mem_offsets) const {
    internal_call_preamble();

    h->mov(h->rbp, reinterpret_cast<uint64_t>(BrgemmKernelExecutor::execute));
    auto reserved_stack_size = sizeof(BrgemmKernelExecutor::call_args);
    // Reserve memory on the stack
    h->sub(h->rsp, reserved_stack_size);

    Xbyak::Reg64 aux_reg = [this, &mem_ptrs_idxs]() {
        std::set<size_t> used(mem_ptrs_idxs.begin(), mem_ptrs_idxs.end());
        std::vector<Xbyak::Reg64> spilled_gprs {h->r8, h->r9, h->r10, h->r11, h->r12, h->r13, h->r14, h->r15,
                                                h->rax, h->rbx, h->rcx, h->rdx, h->rdi, h->rsi, h->rbp};
        for (const auto& reg : spilled_gprs)
            if (used.count(reg.getIdx()) == 0)
                return reg;
        OV_CPU_JIT_EMITTER_THROW("Failed to allocate aux register");
    }();

    auto write_addr_on_stack = [&](size_t arg_offset, Reg64 addr, size_t addr_offset, size_t buffer_id) {
        const auto stack_frame = h->qword[h->rsp + arg_offset];
        h->mov(aux_reg, addr);
        if (snippets::utils::is_dynamic_value(addr_offset))
            h->add(aux_reg,  h->ptr[abi_param1 + GET_OFF(buffer_offsets) + buffer_id * sizeof(size_t)]);
        else if (addr_offset != 0)
            h->add(aux_reg, addr_offset);
        h->mov(stack_frame, aux_reg);
    };
    const std::vector<size_t> brgemm_args_offsets {GET_OFF_BRGEMM_ARGS(A), GET_OFF_BRGEMM_ARGS(B), GET_OFF_BRGEMM_ARGS(C),
                                                   GET_OFF_BRGEMM_ARGS(scratch)};
    const auto& mem_ptrs = utils::transform_idxs_to_regs(mem_ptrs_idxs);
    for (size_t i = 0; i < mem_ptrs.size(); i++)
        write_addr_on_stack(brgemm_args_offsets[i], mem_ptrs[i], mem_offsets[i], m_buffer_ids[i]);

    // No scratchpad => need to write nullptr manually
    if (mem_ptrs.size() < 4)
        h->mov(h->qword[h->rsp + brgemm_args_offsets.back()], reinterpret_cast<uintptr_t>(nullptr));

    // abi_param1 always contains jit_snippets_call_args which has amx tile config for each thread
    h->lea(h->r10, h->ptr[abi_param1 + GET_OFF(amx_tile_config)]);
    h->mov(h->qword[h->rsp + GET_OFF_BRGEMM_ARGS(amx_tile_config)], h->r10);

    h->mov(abi_param1, reinterpret_cast<uintptr_t>(m_kernel_executor.get()));
    h->mov(abi_param2, h->rsp);

    internal_call_rsp_align();
    h->call(h->rbp);
    internal_call_rsp_restore();

    h->add(h->rsp, reserved_stack_size);
    internal_call_postamble();
}

}   // namespace intel_cpu
}   // namespace ov
