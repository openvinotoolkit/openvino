// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_kernel_emitter.hpp"

#include "jit_snippets_emitters.hpp"
#include "snippets/utils/reg_utils.hpp"
#include "utils.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu {

jit_kernel_emitter::jit_kernel_emitter(jit_generator* h,
                                       cpu_isa_t isa,
                                       const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa) {
    const auto kernel = ov::as_type_ptr<snippets::op::Kernel>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(kernel != nullptr, "invoked with invalid op argument");
    OV_CPU_JIT_EMITTER_ASSERT(!kernel->region->empty(), "invoked with empty body");
    body = kernel->region;
    jcp = *reinterpret_cast<const jit_snippets_compile_args*>(kernel->compile_params);
    const auto& parameters = body->get_parameters();
    const auto& results = body->get_results();
    const auto& buffers = body->get_buffers();
    num_inputs = parameters.size();
    num_outputs = results.size();
    std::vector<snippets::Reg> data_ptr_regs;
    data_ptr_regs.reserve(num_inputs + num_outputs);
    for (const auto& param : parameters) {
        data_ptr_regs.push_back(param->get_output_port_descriptor(0)->get_reg());
    }
    for (const auto& result : results) {
        data_ptr_regs.push_back(result->get_input_port_descriptor(0)->get_reg());
    }

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
                                        const std::vector<size_t>& pool_gpr_idxs) const {
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

void jit_kernel_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    h->preamble();

    std::set<snippets::Reg> available_gpr;
    std::set<snippets::Reg> available_vec;
    auto reg_type = snippets::RegType::gpr;
    auto convert = [&reg_type](size_t i) -> snippets::Reg {
        return {reg_type, i};
    };
    std::transform(aux_gpr_idxs.begin(),
                   aux_gpr_idxs.end(),
                   std::inserter(available_gpr, available_gpr.begin()),
                   convert);
    // Note: data_ptr regs are globally live, so it makes no sense to keep them in the pool
    for (auto idx : data_ptr_regs_idx) {
        available_gpr.erase({snippets::RegType::gpr, idx});
    }
    reg_type = snippets::RegType::vec;
    std::transform(aux_vec_idxs.begin(),
                   aux_vec_idxs.end(),
                   std::inserter(available_vec, available_vec.begin()),
                   convert);

    auto data_ptr_regs = utils::transform_idxs_to_regs(data_ptr_regs_idx);

    auto get_expected_reg_types =
        [](const std::shared_ptr<jit_emitter>& emitter) -> std::pair<snippets::RegType, snippets::RegType> {
        switch (emitter->get_in_out_type()) {
        case emitter_in_out_map::gpr_to_vec:
            return {snippets::RegType::gpr, snippets::RegType::vec};
        case emitter_in_out_map::gpr_to_gpr:
            return {snippets::RegType::gpr, snippets::RegType::gpr};
        case emitter_in_out_map::vec_to_gpr:
            return {snippets::RegType::vec, snippets::RegType::gpr};
        case emitter_in_out_map::vec_to_vec:
            return {snippets::RegType::vec, snippets::RegType::vec};
        default:
            OV_CPU_JIT_EMITTER_THROW("Unsupported emitter_in_out_map instance");
        }
    };
    std::vector<Xbyak::Reg64> aux_tmp_regs{};
    if (!available_gpr.empty()) {
        aux_tmp_regs.emplace_back(available_gpr.begin()->idx);
    }
    init_data_pointers(utils::transform_idxs_to_regs(in), data_ptr_regs, aux_tmp_regs);
    for (const auto& expression : *body) {
        const auto reg_info = expression->get_reg_info();
        const auto& emitter = std::dynamic_pointer_cast<jit_emitter>(expression->get_emitter());
        OV_CPU_JIT_EMITTER_ASSERT(emitter, "Unexpected emitter type");
        auto expected_in_type = snippets::RegType::undefined;
        auto expected_out_type = snippets::RegType::undefined;
        const auto& node = expression->get_node();
        // Note: A few operations are allowed to have mixed register types on their inputs (or outputs) => skip
        // validation here
        if (!ov::is_type_any_of<snippets::op::LoopEnd, snippets::op::RegSpillBase>(node) &&
            !std::dynamic_pointer_cast<jit_nop_emitter>(emitter)) {
            std::tie(expected_in_type, expected_out_type) = get_expected_reg_types(emitter);
        }
        // Note: live regs = regs live on input of the expression. We also need to exclude output regs from the pool
        auto live_regs = expression->get_live_regs();
        for (auto r : reg_info.second) {
            live_regs.insert(r);
        }
        std::vector<snippets::Reg> pool_gp_reg;
        std::vector<snippets::Reg> pool_vec_reg;
        std::set_difference(available_gpr.begin(),
                            available_gpr.end(),
                            live_regs.begin(),
                            live_regs.end(),
                            std::back_inserter(pool_gp_reg));
        std::set_difference(available_vec.begin(),
                            available_vec.end(),
                            live_regs.begin(),
                            live_regs.end(),
                            std::back_inserter(pool_vec_reg));
        auto in_regs = snippets::utils::transform_snippets_regs_to_idxs(reg_info.first, expected_in_type);
        auto out_regs = snippets::utils::transform_snippets_regs_to_idxs(reg_info.second, expected_out_type);
        auto gpr_pool = snippets::utils::transform_snippets_regs_to_idxs(pool_gp_reg);
        auto vec_pool = snippets::utils::transform_snippets_regs_to_idxs(pool_vec_reg);
        emitter->emit_code(in_regs, out_regs, vec_pool, gpr_pool);
    }

    h->postamble();
}

jit_kernel_static_emitter::jit_kernel_static_emitter(dnnl::impl::cpu::x64::jit_generator* h,
                                                     dnnl::impl::cpu::x64::cpu_isa_t isa,
                                                     const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_kernel_emitter(h, isa, expr) {
    const auto kernel = ov::as_type_ptr<snippets::op::KernelStatic>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(kernel != nullptr, "expects KernelStatic expression");
    jcp = *reinterpret_cast<const jit_snippets_compile_args*>(kernel->compile_params);
    master_shape = jcp.exec_domain;
    data_offsets = jcp.data_offsets;
    OV_CPU_JIT_EMITTER_ASSERT(data_offsets.size() == num_inputs + num_outputs, "Incompatible count of data offsets!");
    OV_CPU_JIT_EMITTER_ASSERT(data_offsets.front().size() == master_shape.size(), "Incompatible rank of data offsets!");
}

void jit_kernel_static_emitter::init_data_pointers(const std::vector<Xbyak::Reg64>& arg_regs,
                                                   const std::vector<Xbyak::Reg64>& data_ptr_regs,
                                                   const std::vector<Xbyak::Reg64>& aux_gprs) const {
    OV_CPU_JIT_EMITTER_ASSERT(arg_regs.size() == 2, "Invalid arg regs size");
    Xbyak::Reg64 reg_runtime_params = arg_regs[0];
    Xbyak::Reg64 reg_indexes = arg_regs[1];

    const auto num_params = num_inputs + num_outputs;
    // Note that we don't need offset for the last dim, since it's handled directly by Tile emitter
    const size_t offset_rank = master_shape.size() - 1;

    // master_shape size must be valid in both static and dynamic cases
    std::function<void(Reg64, const std::vector<size_t>&, Reg64)> init_ptr_with_offset;
    init_ptr_with_offset = [&](Reg64 pointer, const std::vector<size_t>& offsets, Reg64 reg_tmp) {
        for (size_t j = 0; j < offset_rank; j++) {
            if (master_shape[j] != 1 && offsets[j] != 0) {
                h->mov(reg_tmp, offsets[j]);
                h->imul(reg_tmp, h->ptr[reg_indexes + j * sizeof(size_t)]);
                h->add(pointer, reg_tmp);
            }
        }
    };
    const bool last_iter_explicitly = aux_gprs.empty();
    Reg64 reg_tmp = last_iter_explicitly ? data_ptr_regs[num_params - 1] : *aux_gprs.begin();
    // Vector "data_ptr_regs" is sorted by abstract regs.
    // It means that the vector contains the physical registers in order [src, .., src, dst, .., dst, buffer]
    // So we can initialize buffer register firstly as last value of vector "data_ptr_regs"
    // NOTE: Snippets Buffer Scratchpad has the common data pointer for all Buffers (even with different ID).
    //       The accessing memory is covered by correct offsets in each Buffer and the corresponding MemoryAccess ops
    for (size_t i = 0; i < num_unique_buffers; ++i) {
        h->mov(data_ptr_regs[num_params + i], h->ptr[reg_runtime_params + GET_OFF(buffer_scratchpad_ptr)]);
    }
    size_t i = 0;
    for (; i < num_params - last_iter_explicitly; i++) {
        if (i < num_inputs) {
            h->mov(data_ptr_regs[i], h->ptr[reg_runtime_params + GET_OFF(src_ptrs) + i * sizeof(void*)]);
        } else {
            h->mov(data_ptr_regs[i], h->ptr[reg_runtime_params + GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*)]);
        }
        init_ptr_with_offset(data_ptr_regs[i], data_offsets[i], reg_tmp);
    }
    // A rare case when num_params is maximal, so we have no spare gprs
    // Note that we need to push-pop runtime params because some kernels might need them even in the static case
    // (e.g. brgemm emitter for amx tile configuration access)
    if (last_iter_explicitly) {
        h->push(reg_runtime_params);
        h->mov(data_ptr_regs[i], h->ptr[reg_runtime_params + GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*)]);
        reg_tmp = reg_runtime_params;
        // can corrupt reg_runtime_params, since we won't use it anymore
        init_ptr_with_offset(data_ptr_regs[i], data_offsets[i], reg_tmp);
        h->pop(reg_runtime_params);
    }
}

jit_kernel_dynamic_emitter::jit_kernel_dynamic_emitter(dnnl::impl::cpu::x64::jit_generator* h,
                                                       dnnl::impl::cpu::x64::cpu_isa_t isa,
                                                       const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_kernel_emitter(h, isa, expr) {
    OV_CPU_JIT_EMITTER_ASSERT(ov::is_type<snippets::op::KernelDynamic>(expr->get_node()),
                              "expects KernelDynamic expression");
}

void jit_kernel_dynamic_emitter::init_data_pointers(const std::vector<Xbyak::Reg64>& arg_regs,
                                                    const std::vector<Xbyak::Reg64>& data_ptr_regs,
                                                    const std::vector<Xbyak::Reg64>& aux_gprs) const {
    OV_CPU_JIT_EMITTER_ASSERT(arg_regs.size() == 1, "Invalid arg regs size");
    Xbyak::Reg64 reg_runtime_params = arg_regs[0];

    const auto num_params = num_inputs + num_outputs;
    for (size_t i = 0; i < num_unique_buffers; ++i) {
        h->mov(data_ptr_regs[num_params + i], h->ptr[reg_runtime_params + GET_OFF(buffer_scratchpad_ptr)]);
    }
    for (size_t i = 0; i < num_params; i++) {
        if (i < num_inputs) {
            h->mov(data_ptr_regs[i], h->ptr[reg_runtime_params + GET_OFF(src_ptrs) + i * sizeof(void*)]);
        } else {
            h->mov(data_ptr_regs[i], h->ptr[reg_runtime_params + GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*)]);
        }
    }
}

}  // namespace ov::intel_cpu
