// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_kernel_emitter.hpp"

#include "snippets/utils/utils.hpp"
#include "utils.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {

jit_kernel_emitter::jit_kernel_emitter(jit_generator* h, cpu_isa_t isa, const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa), reg_runtime_params_idx(abi_param1.getIdx()) {
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
    for (const auto& param : parameters)
        mem_access_exprs.push_back(param);
    for (const auto& result : results)
        mem_access_exprs.push_back(result);

    std::set<size_t> unique_buffers;
    for (const auto& buffer_expr : buffers) {
        const auto buffer_reg_group = buffer_expr->get_reg_group();
        if (unique_buffers.count(buffer_reg_group) == 0) {
            mem_access_exprs.push_back(buffer_expr);
            unique_buffers.insert(buffer_reg_group);
        }
    }

    using ExprSet = std::unordered_set<snippets::lowered::ExpressionPtr>;
    const ExprSet params_set(parameters.cbegin(), parameters.cend());
    const ExprSet results_set(results.cbegin(), results.cend());
    const ExprSet buffers_set(buffers.cbegin(), buffers.cend());
    for (const auto& expr : *body) {
        if (params_set.count(expr) == 0 && results_set.count(expr) == 0 && buffers_set.count(expr) == 0)
            general_exprs.emplace_back(expr);
    }
    num_unique_buffers = unique_buffers.size();
}

void jit_kernel_emitter::init_reg_pools(const std::set<size_t>& gpr_blacklist, const std::set<size_t>& vec_blacklist) {
    gp_regs_pool.resize(16);
    vec_regs_pool.resize(16);
    // It's easier to remove the last item during mapping, so fill descending to map ascending
    for (size_t i = 0; i < 16; i++)
        gp_regs_pool[i] = vec_regs_pool[i] = 15 - i;
    auto remove_regs_from_pool = [](std::vector<size_t>& pool, const std::set<size_t>& to_remove) {
        // It's important to keep the order of other elements
        pool.erase(std::remove_if(pool.begin(), pool.end(),
                                  [&](size_t x) {return to_remove.count(x) != 0;}), pool.end());
    };
    // Reserve stack base and pointer for push(...) and pop(...) operations
    std::set<size_t> gprs_blacklist_extended{Xbyak::Operand::RSP, Xbyak::Operand::RBP};
    gprs_blacklist_extended.insert(gpr_blacklist.begin(), gpr_blacklist.end());
    // Reserve abi_param1 and abi_param2, since they'll be used to pass runtime call args to kernel
    remove_regs_from_pool(gp_regs_pool, gprs_blacklist_extended);
    remove_regs_from_pool(vec_regs_pool, vec_blacklist);
}

void jit_kernel_emitter::emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out,
                                   const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void jit_kernel_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.empty() && out.empty(), ": expects 0 registers on input and output");
    const auto num_params = num_inputs + num_outputs + num_unique_buffers;
    // The number of used gpr may be >= num_params since LoopBegin+LoopEnd could also use gpr to store work_amount
    OV_CPU_JIT_EMITTER_ASSERT(data_ptr_regs_idx.size() == num_params,
                              "number of inputs and outputs is inconsistent with the number of allocated registers ", num_params,
                              " data_ptr_regs_idx.size() = ", data_ptr_regs_idx.size());
}

void jit_kernel_emitter::init_body_regs(const std::set<size_t>& kernel_regs,
                                        const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) {
    // Initialize pools of gp and vec registers
    // Reserve kernel regs (abi_param1 and, if there is, abi_param2), since they'll be used to pass runtime call args to kernel
    init_reg_pools(kernel_regs, {});

    mapping_info gpr_map_pool({}, gp_regs_pool);
    mapping_info vec_map_pool({}, vec_regs_pool);

    // Note that we can't use kernel_regs to store data pointers because
    // these regs are used to calculate offsets for the data pointers
    map_abstract_registers(gpr_map_pool, vec_map_pool, mem_access_exprs);
    for (const auto& abstract_to_physical : gpr_map_pool.first)
        data_ptr_regs_idx.push_back(abstract_to_physical.second);

    gpr_map_pool.second.insert(gpr_map_pool.second.end(), pool_gpr_idxs.cbegin(), pool_gpr_idxs.cend());
    vec_map_pool.second.insert(vec_map_pool.second.end(), pool_vec_idxs.cbegin(), pool_vec_idxs.cend());
    map_abstract_registers(gpr_map_pool, vec_map_pool, general_exprs);
}

void jit_kernel_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    h->preamble();

    auto data_ptr_regs = utils::transform_idxs_to_regs(data_ptr_regs_idx);

    init_data_pointers(data_ptr_regs);
    for (const auto& expression : *body) {
        const auto reg_info = expression->get_reg_info();
        auto in_regs = utils::transform_snippets_regs_to_idxs(reg_info.first);
        auto out_regs = utils::transform_snippets_regs_to_idxs(reg_info.second);
        const auto& emitter = expression->get_emitter();
        emitter->emit_code(in_regs, out_regs, vec_regs_pool, gp_regs_pool);
    }

    h->postamble();
}

jit_kernel_static_emitter::jit_kernel_static_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                                     const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_kernel_emitter(h, isa, expr), reg_indexes_idx(abi_param2.getIdx()) {
    const auto kernel = ov::as_type_ptr<snippets::op::KernelStatic>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(kernel != nullptr, "expectes KernelStatic expression");
    jcp = *reinterpret_cast<const jit_snippets_compile_args*>(kernel->compile_params);
    master_shape = jcp.exec_domain;
    data_offsets = jcp.data_offsets;
    OV_CPU_JIT_EMITTER_ASSERT(data_offsets.size() == num_inputs + num_outputs, "Incompatible count of data offsets!");
    OV_CPU_JIT_EMITTER_ASSERT(data_offsets.front().size() == master_shape.size(), "Incompatible rank of data offsets!");

    // - Reserve abi_param1 and abi_param2, since they'll be used to pass runtime call args to kernel
    // - However we can use reg_indexes_idx for non memory access operations
    //   since we won't need them after offsets calculation
    init_body_regs({reg_indexes_idx, reg_runtime_params_idx}, {}, {reg_indexes_idx});
}

void jit_kernel_static_emitter::init_data_pointers(const std::vector<Xbyak::Reg64>& data_ptr_regs) const {
    Xbyak::Reg64 reg_indexes = Xbyak::Reg64(static_cast<int>(reg_indexes_idx));
    Xbyak::Reg64 reg_runtime_params = Xbyak::Reg64(static_cast<int>(reg_runtime_params_idx));

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
    const auto spare_corruptable_gpr = std::find_if(gp_regs_pool.begin(), gp_regs_pool.end(),
                                                   [this](size_t reg) {
                                                        return reg != reg_indexes_idx && reg != reg_runtime_params_idx;
                                                   });
    const bool last_iter_explicitly = spare_corruptable_gpr == gp_regs_pool.end();
    Reg64 reg_tmp = last_iter_explicitly ? data_ptr_regs[num_params - 1] : Reg64(static_cast<int>(*spare_corruptable_gpr));
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
        if (i < num_inputs)
            h->mov(data_ptr_regs[i], h->ptr[reg_runtime_params + GET_OFF(src_ptrs) + i * sizeof(void*)]);
        else
            h->mov(data_ptr_regs[i], h->ptr[reg_runtime_params + GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*)]);
        init_ptr_with_offset(data_ptr_regs[i], data_offsets[i], reg_tmp);
    }
    // a rare case when num_params is maximal, so we have no spare gprs
    // * Static case: we can use reg_runtime_params as the last reg_tmp for the last iteration (and corrupt it), since
    //     it won't be used anymore
    // * Dynamic case: we will need reg_runtime_params to pass runtime args to LoopScheduler, so we have to
    //     push a reg on the stack, and restore it value afterward
    if (last_iter_explicitly) {
        h->mov(data_ptr_regs[i], h->ptr[reg_runtime_params + GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*)]);
        reg_tmp = reg_runtime_params;
        // can corrupt reg_runtime_params, since we won't use it anymore
        init_ptr_with_offset(data_ptr_regs[i], data_offsets[i], reg_tmp);
    }
}

jit_kernel_dynamic_emitter::jit_kernel_dynamic_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                                       const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_kernel_emitter(h, isa, expr) {
    const auto kernel = ov::as_type_ptr<snippets::op::KernelDynamic>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(kernel, "expectes KernelDynamic expression");

    // - Reserve abi_param1, since it wll be used to pass runtime call args to all dynamic emitters that needs runtime args
    // - We cannot assign this register to the body emitters since runtime params MUST be valid during whole execution
    //   for all dynamic emitters
    init_body_regs({reg_runtime_params_idx});
}

void jit_kernel_dynamic_emitter::init_data_pointers(const std::vector<Xbyak::Reg64>& data_ptr_regs) const {
    Xbyak::Reg64 reg_runtime_params = Xbyak::Reg64(static_cast<int>(reg_runtime_params_idx));

    const auto num_params = num_inputs + num_outputs;
    for (size_t i = 0; i < num_unique_buffers; ++i) {
        h->mov(data_ptr_regs[num_params + i], h->ptr[reg_runtime_params + GET_OFF(buffer_scratchpad_ptr)]);
    }
    for (size_t i = 0; i < num_params; i++) {
        if (i < num_inputs)
            h->mov(data_ptr_regs[i], h->ptr[reg_runtime_params + GET_OFF(src_ptrs) + i * sizeof(void*)]);
        else
            h->mov(data_ptr_regs[i], h->ptr[reg_runtime_params + GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*)]);
    }
}

}   // namespace intel_cpu
}   // namespace ov
