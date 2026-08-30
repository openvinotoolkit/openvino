// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_brgemm_emitter.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <vector>

#include "emitters/plugin/riscv64/jit_context_helpers.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/snippets/riscv64/jit_binary_call_emitter.hpp"
#include "emitters/snippets/riscv64/kernel_executors/brgemm.hpp"
#include "emitters/snippets/riscv64/utils.hpp"
#include "emitters/snippets/utils/utils.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/type.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/riscv64/op/brgemm_cpu.hpp"
#include "transformations/snippets/riscv64/op/brgemm_utils.hpp"
#include "xbyak_riscv/xbyak_riscv.hpp"

namespace ov::intel_cpu::riscv64 {

jit_brgemm_emitter::jit_brgemm_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                       ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                       const ov::snippets::lowered::ExpressionPtr& expr,
                                       const snippets::KernelExecutorTablePtr& kernel_table,
                                       const ov::intel_cpu::MultiCacheWeakPtr& compiled_kernel_cache)
    : jit_binary_call_emitter(host, host_isa, expr->get_live_regs()) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;

    const auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(brgemm, "Expected BrgemmCPU node");
    m_kernel_executor =
        kernel_table->register_kernel<BrgemmKernelExecutor>(expr,
                                                            compiled_kernel_cache,
                                                            BrgemmKernelConfig{brgemm->get_input_element_type(0)});

    m_memory_offsets = {brgemm->get_offset_a(), brgemm->get_offset_b(), brgemm->get_offset_c()};
    m_buffer_ids = {ov::intel_cpu::utils::get_buffer_cluster_id(expr->get_input_port(0)),
                    ov::intel_cpu::utils::get_buffer_cluster_id(expr->get_input_port(1)),
                    ov::intel_cpu::utils::get_buffer_cluster_id(expr->get_output_port(0))};
}

std::set<std::vector<element::Type>> jit_brgemm_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    std::set<std::vector<element::Type>> result;
    if (brgemm_utils::is_fp32_supported()) {
        result.insert({element::f32, element::f32});
    }
    if (brgemm_utils::is_bf16_supported()) {
        result.insert({element::bf16, element::bf16});
    }
    if (brgemm_utils::is_fp16_supported()) {
        result.insert({element::f16, element::f16});
    }
    return result;
}

void jit_brgemm_emitter::validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == 2, "BrgemmCPU expects two input registers");
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 1, "BrgemmCPU expects one output register");
}

void jit_brgemm_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    validate_arguments(in, out);
    const std::vector<size_t> memory_ptr_indices{in[0], in[1], out[0]};
    init_binary_call_regs(2, memory_ptr_indices);
    binary_call_preamble();

    const auto frame_size = rnd_up(sizeof(BrgemmKernelExecutor::call_args), sp_alignment);
    utils::sub_sp(*h, frame_size);

    const auto& auxiliary = get_call_address_reg();
    const auto memory_ptrs = utils::transform_idxs_to_regs(memory_ptr_indices);
    constexpr int32_t argument_offsets[] = {
        static_cast<int32_t>(offsetof(BrgemmKernelExecutor::call_args, A)),
        static_cast<int32_t>(offsetof(BrgemmKernelExecutor::call_args, B)),
        static_cast<int32_t>(offsetof(BrgemmKernelExecutor::call_args, C)),
    };

    for (size_t i = 0; i < memory_ptrs.size(); ++i) {
        if (snippets::utils::is_dynamic_value(m_memory_offsets[i])) {
            const auto runtime_offset = GET_OFF(buffer_offsets) + m_buffer_ids[i] * sizeof(size_t);
            h->ld(auxiliary, Xbyak_riscv::a0, static_cast<int32_t>(runtime_offset));
            h->add(auxiliary, memory_ptrs[i], auxiliary);
            h->sd(auxiliary, Xbyak_riscv::sp, argument_offsets[i]);
        } else if (m_memory_offsets[i] == 0) {
            h->sd(memory_ptrs[i], Xbyak_riscv::sp, argument_offsets[i]);
        } else {
            h->uni_li(auxiliary, m_memory_offsets[i]);
            h->add(auxiliary, memory_ptrs[i], auxiliary);
            h->sd(auxiliary, Xbyak_riscv::sp, argument_offsets[i]);
        }
    }

    h->uni_li(auxiliary, reinterpret_cast<uintptr_t>(BrgemmKernelExecutor::execute));
    h->uni_li(Xbyak_riscv::a0, reinterpret_cast<uintptr_t>(m_kernel_executor.get()));
    h->mv(Xbyak_riscv::a1, Xbyak_riscv::sp);
    h->jalr(Xbyak_riscv::ra, auxiliary);

    utils::add_sp(*h, frame_size);
    binary_call_postamble();
}

}  // namespace ov::intel_cpu::riscv64
