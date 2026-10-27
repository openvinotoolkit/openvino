// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_kernel_emitter.hpp"

#include <cstddef>
#include <set>
#include <vector>

#include "emitters/plugin/riscv64/jit_emitter.hpp"
#include "emitters/snippets/common/jit_kernel_emitter_utils.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/utils.hpp"
#include "jit_snippets_emitters.hpp"
#include "nodes/kernels/riscv64/cpu_isa_traits.hpp"
#include "nodes/kernels/riscv64/jit_generator.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "snippets/emitter.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/kernel.hpp"
#include "snippets/utils/reg_utils.hpp"
#include "utils.hpp"
#include "xbyak_riscv/xbyak_riscv.hpp"

using namespace Xbyak_riscv;

namespace ov::intel_cpu::riscv64 {

jit_kernel_emitter::jit_kernel_emitter(jit_generator_t* h,
                                       cpu_isa_t isa,
                                       const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa) {
    const auto kernel = ov::as_type_ptr<snippets::op::Kernel>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(kernel != nullptr, "jit_kernel_emitter invoked with invalid op argument");
    OV_CPU_JIT_EMITTER_ASSERT(!kernel->region->empty(), "jit_kernel_emitter invoked with empty body");
    body = kernel->region;
    jcp = *reinterpret_cast<const jit_snippets_compile_args*>(kernel->compile_params);
    const auto& parameters = body->get_parameters();
    const auto& results = body->get_results();
    const auto& buffers = body->get_buffers();
    std::vector<snippets::Reg> data_ptr_regs;
    for (const auto& param : parameters) {
        const auto& reg = param->get_output_port_descriptor(0)->get_reg();
        if (!reg.is_address()) {
            data_ptr_regs.push_back(reg);
        }
    }
    num_inputs = data_ptr_regs.size();
    for (const auto& result : results) {
        data_ptr_regs.push_back(result->get_input_port_descriptor(0)->get_reg());
    }
    num_outputs = data_ptr_regs.size() - num_inputs;

    std::set<size_t> unique_buffers;
    for (const auto& buffer_expr : buffers) {
        const auto buffer_reg_group = buffer_expr->get_reg_group();
        if (unique_buffers.count(buffer_reg_group) == 0) {
            data_ptr_regs.push_back(buffer_expr->get_output_port_descriptor(0)->get_reg());
            unique_buffers.insert(buffer_reg_group);
        }
    }
    num_unique_buffers = unique_buffers.size();
    data_ptr_regs_idx = snippets::utils::transform_snippets_regs_to_idxs(data_ptr_regs, snippets::RegType::gpr);
}

void jit_kernel_emitter::emit_code_impl(const std::vector<size_t>& in,
                                        const std::vector<size_t>& out,
                                        const std::vector<size_t>& pool_vec_idxs,
                                        const std::vector<size_t>& pool_gpr_idxs,
                                        [[maybe_unused]] const std::vector<size_t>& pool_fp_gpr_idxs) const {
    validate_arguments(in, out);
    aux_vec_idxs = pool_vec_idxs;
    aux_gpr_idxs = pool_gpr_idxs;
    emit_impl(in, out);
}

void jit_kernel_emitter::validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == get_inputs_num() && out.empty(),
                              "Unexpected number of input/output arguments");
    const auto num_params = num_inputs + num_outputs + num_unique_buffers;
    // The number of used gpr may be >= num_params since LoopBegin+LoopEnd could also use gpr to store work_amount
    OV_CPU_JIT_EMITTER_ASSERT(data_ptr_regs_idx.size() == num_params,
                              "Number of inputs and outputs is inconsistent with the number of allocated registers ",
                              num_params,
                              " data_ptr_regs_idx.size() = ",
                              data_ptr_regs_idx.size());
}

void jit_kernel_emitter::emit_impl(const std::vector<size_t>& in,
                                   [[maybe_unused]] const std::vector<size_t>& out) const {
    h->preamble();
    const auto register_pools =
        kernel_emitter_utils::make_register_pools(aux_gpr_idxs, aux_vec_idxs, data_ptr_regs_idx);
    auto data_ptr_regs = utils::transform_idxs_to_regs(data_ptr_regs_idx);
    init_data_pointers(utils::transform_idxs_to_regs(in), data_ptr_regs);

    kernel_emitter_utils::emit_kernel_body<jit_emitter, jit_nop_emitter>(*body, register_pools);

    h->postamble();
}

jit_kernel_static_emitter::jit_kernel_static_emitter(jit_generator_t* h,
                                                     cpu_isa_t isa,
                                                     const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_kernel_emitter(h, isa, expr) {
    const auto kernel = ov::as_type_ptr<snippets::op::KernelStatic>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(kernel != nullptr, "jit_kernel_static_emitter expects KernelStatic expression");
    jcp = *reinterpret_cast<const jit_snippets_compile_args*>(kernel->compile_params);
    master_shape = jcp.exec_domain;
    data_offsets = jcp.data_offsets;
    OV_CPU_JIT_EMITTER_ASSERT(data_offsets.size() == num_inputs + num_outputs, "Incompatible count of data offsets!");
    OV_CPU_JIT_EMITTER_ASSERT(!data_offsets.empty() && data_offsets.front().size() == master_shape.size(),
                              "Incompatible rank of data offsets!");
}

void jit_kernel_static_emitter::init_data_pointers(const std::vector<Xbyak_riscv::Reg>& arg_regs,
                                                   const std::vector<Xbyak_riscv::Reg>& data_ptr_regs,
                                                   const std::vector<Xbyak_riscv::Reg>& aux_gprs) const {
    OV_CPU_JIT_EMITTER_ASSERT(arg_regs.size() == 2, "Invalid arg regs size");
    auto reg_runtime_params = arg_regs[0];
    auto reg_indexes = arg_regs[1];

    const auto num_params = num_inputs + num_outputs;
    // Note that we don't need offset for the last dim, since it's handled directly by Tile emitter
    const size_t offset_rank = master_shape.size() - 1;

    // helper: pointer += offsets[j] * indexes[j]
    // uses two temporaries to avoid clobbering the offset constant while loading the index
    auto init_ptr_with_offset = [&](Xbyak_riscv::Reg pointer,
                                    const std::vector<size_t>& offsets,
                                    Xbyak_riscv::Reg tmp0,
                                    Xbyak_riscv::Reg tmp1) {
        for (size_t j = 0; j < offset_rank; j++) {
            if (master_shape[j] != 1 && offsets[j] != 0) {
                // tmp0 = offsets[j]
                h->uni_li(tmp0, offsets[j]);
                // tmp1 = address of index[j]
                h->uni_li(tmp1, j * sizeof(size_t));
                h->add(tmp1, reg_indexes, tmp1);
                // tmp1 = load index[j]
                h->ld(tmp1, tmp1, 0);
                // tmp0 *= tmp1
                h->mul(tmp0, tmp0, tmp1);
                // pointer += tmp0
                h->add(pointer, pointer, tmp0);
            }
        }
    };

    // choose tmp regs
    Xbyak_riscv::Reg tmp0 = !aux_gprs.empty() ? aux_gprs[0] : Xbyak_riscv::t0;
    Xbyak_riscv::Reg tmp1 = aux_gprs.size() > 1 ? aux_gprs[1] : Xbyak_riscv::t1;

    // Initialize buffer scratchpad pointers
    for (size_t i = 0; i < num_unique_buffers; ++i) {
        Xbyak_riscv::Reg addr = tmp0;
        h->uni_li(addr, GET_OFF(buffer_scratchpad_ptr));
        h->add(addr, reg_runtime_params, addr);
        h->ld(data_ptr_regs[num_params + i], addr, 0);
    }

    // Load input/output pointers and apply static offsets
    for (size_t i = 0; i < num_params; i++) {
        Xbyak_riscv::Reg addr = tmp0;
        if (i < num_inputs) {
            h->uni_li(addr, GET_OFF(src_ptrs) + i * sizeof(void*));
        } else {
            h->uni_li(addr, GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*));
        }
        h->add(addr, reg_runtime_params, addr);
        h->ld(data_ptr_regs[i], addr, 0);
        init_ptr_with_offset(data_ptr_regs[i], data_offsets[i], tmp0, tmp1);
    }
}

jit_kernel_dynamic_emitter::jit_kernel_dynamic_emitter(jit_generator_t* h,
                                                       cpu_isa_t isa,
                                                       const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_kernel_emitter(h, isa, expr) {
    OV_CPU_JIT_EMITTER_ASSERT(ov::is_type<snippets::op::KernelDynamic>(expr->get_node()),
                              "jit_kernel_dynamic_emitter expects KernelDynamic expression");
}

void jit_kernel_dynamic_emitter::init_data_pointers(
    const std::vector<Xbyak_riscv::Reg>& arg_regs,
    const std::vector<Xbyak_riscv::Reg>& data_ptr_regs,
    [[maybe_unused]] const std::vector<Xbyak_riscv::Reg>& aux_gprs) const {
    OV_CPU_JIT_EMITTER_ASSERT(arg_regs.size() == 1, "Invalid arg regs size");
    auto reg_runtime_params = arg_regs[0];

    const auto num_params = num_inputs + num_outputs;
    for (size_t i = 0; i < num_unique_buffers; ++i) {
        Xbyak_riscv::Reg addr = Xbyak_riscv::t0;
        h->uni_li(addr, GET_OFF(buffer_scratchpad_ptr));
        h->add(addr, reg_runtime_params, addr);
        h->ld(data_ptr_regs[num_params + i], addr, 0);
    }
    for (size_t i = 0; i < num_params; i++) {
        Xbyak_riscv::Reg addr = aux_gprs.empty() ? Xbyak_riscv::t0 : aux_gprs.front();
        if (i < num_inputs) {
            h->uni_li(addr, GET_OFF(src_ptrs) + i * sizeof(void*));
        } else {
            h->uni_li(addr, GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*));
        }
        h->add(addr, reg_runtime_params, addr);
        h->ld(data_ptr_regs[i], addr, 0);
    }
}

}  // namespace ov::intel_cpu::riscv64
