// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_brgemm_emitter.hpp"

#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include <cpu/x64/brgemm/brgemm.hpp>
#include <cpu/x64/amx_tile_configure.hpp>
#include "snippets/utils.hpp"

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

    m_with_scratch = brgemm_node->is_with_scratchpad();

    const auto& brg0Prc = brgemm_node->get_input_element_type(0);
    const auto& brg1Prc = brgemm_node->get_input_element_type(1);
    auto kernel_config = std::make_shared<BrgemmKernelConfig>(brg0Prc, brg1Prc,
                                                            brgemm_node->get_beta(),
                                                            brgemm_node->is_amx(),
                                                            brgemm_node->is_with_compensations());
    m_kernel_executor = kernel_table->register_kernel<BrgemmKernelExecutor>(expr,
                                                                            compiled_kernel_cache,
                                                                            kernel_config);
    // Note: even if the Brgemm node is dynamic, the first shapeInfer and RuntimeConfigurator::update()
    // are performed before the BrgemmKernelExecutor registration. So we have to trigger update() manually
    // for both static and the 1st dynamic shapes.
    OV_CPU_JIT_EMITTER_ASSERT(!snippets::utils::is_dynamic_vdims(expr->get_input_port_descriptor(0)->get_shape()) &&
                              !snippets::utils::is_dynamic_vdims(expr->get_input_port_descriptor(1)->get_shape()),
                              "Jit emitter is called when the shapes are unknown");

    m_load_offset_a = brgemm_node->get_offset_a();
    m_load_offset_b = brgemm_node->get_offset_b();
    m_store_offset_c = brgemm_node->get_offset_c();
    if (m_with_scratch)
        m_load_offset_scratch = brgemm_node->get_offset_scratch();
}

std::set<std::vector<element::Type>> jit_brgemm_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    const auto brgemm = as_type_ptr<ov::intel_cpu::BrgemmCPU>(node);
    OV_CPU_JIT_EMITTER_ASSERT(brgemm, "get_supported_precisions() expects BrgemmCPU node");
    switch (brgemm->get_type()) {
        case BrgemmCPU::Type::Floating:
            return {{element::f32, element::f32}};
        case BrgemmCPU::Type::WithDataRepacking:
            return {{element::u8, element::i8},
                    {element::bf16, element::bf16}};
        case BrgemmCPU::Type::WithCompensations:
            return {{element::i8, element::i8, element::f32}};
        case BrgemmCPU::Type::AMX:
            return {{element::i8, element::i8, element::u8},
                    {element::u8, element::i8, element::u8},
                    {element::bf16, element::bf16, element::u8}};
        default:
            OV_CPU_JIT_EMITTER_THROW("got BrgemmCPU node with unsupported type");
    }
}

void jit_brgemm_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OV_CPU_JIT_EMITTER_ASSERT((m_with_scratch && in.size() == 3) || (!m_with_scratch && in.size() == 2),
                              "expects 3 inputs if there are compensations/wsp");
}

void jit_brgemm_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    validate_arguments(in, out);
    if (host_isa_ == cpu::x64::avx512_core) {
        Xbyak::Reg64 input_0(static_cast<int>(in[0]));
        Xbyak::Reg64 input_1(static_cast<int>(in[1]));
        Xbyak::Reg64 input_2(static_cast<int>(m_with_scratch ? in[2] : 0));  // scratch. Default reg index is 0 if there isn't scratch
        Xbyak::Reg64 output_0(static_cast<int>(out[0]));
        emit_brgemm_kernel_call(input_0,
                                input_1,
                                input_2,
                                output_0,
                                m_load_offset_a,
                                m_load_offset_b,
                                m_load_offset_scratch,
                                m_store_offset_c);
    } else {
        OV_CPU_JIT_EMITTER_THROW("requires at least avx512_core instruction set");
    }
}

void jit_brgemm_emitter::emit_brgemm_kernel_call(Reg64 addr_A, Reg64 addr_B, Reg64 scratch, Reg64 addr_C,
                                                 const size_t in0_kernel_offset, const size_t in1_kernel_offset,
                                                 const size_t in2_kernel_offset, const size_t out0_kernel_offset) const {
    internal_call_preamble();
    h->mov(h->rbp, reinterpret_cast<uint64_t>(BrgemmKernelExecutor::execute));
    auto reserved_stack_size = sizeof(BrgemmKernelExecutor::call_args);
    // Reserve memory on the stack
    h->sub(h->rsp, reserved_stack_size);

    auto write_addr_on_stack = [&](size_t arg_offset, Reg64 addr, size_t addr_offset) {
        const auto stack_frame = h->qword[h->rsp + arg_offset];
        h->mov(stack_frame, addr);
        if (addr_offset) h->add(stack_frame, addr_offset);
    };

    write_addr_on_stack(GET_OFF_BRGEMM_ARGS(A), addr_A, in0_kernel_offset);
    write_addr_on_stack(GET_OFF_BRGEMM_ARGS(B), addr_B, in1_kernel_offset);
    write_addr_on_stack(GET_OFF_BRGEMM_ARGS(C), addr_C, out0_kernel_offset);

    if (m_with_scratch) {
        write_addr_on_stack(GET_OFF_BRGEMM_ARGS(scratch), scratch, in2_kernel_offset);
    } else {
        h->mov(h->qword[h->rsp + GET_OFF_BRGEMM_ARGS(scratch)], reinterpret_cast<uintptr_t>(nullptr));
    }

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
