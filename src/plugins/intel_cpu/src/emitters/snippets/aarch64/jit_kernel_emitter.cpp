// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_kernel_emitter.hpp"

#include "emitters/utils.hpp"
#include "snippets/utils/utils.hpp"

using namespace Xbyak_aarch64;

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using jit_generator = dnnl::impl::cpu::aarch64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::aarch64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

inline static std::vector<XReg> transform_idxs_to_regs(const std::vector<size_t>& idxs) {
    std::vector<XReg> regs(idxs.size(), XReg(0));
    std::transform(idxs.begin(), idxs.end(), regs.begin(), [](size_t idx){return XReg(idx);});
    return regs;
}

inline static std::vector<size_t> transform_snippets_regs_to_idxs(const std::vector<snippets::Reg>& regs) {
    std::vector<size_t> idxs(regs.size());
    std::transform(regs.cbegin(), regs.cend(), idxs.begin(), [](const snippets::Reg& reg) { return reg.idx; });
    return idxs;
}

jit_kernel_emitter::jit_kernel_emitter(jit_generator* h, cpu_isa_t isa, const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa), reg_runtime_params_idx(Operand::X0) {
    const auto kernel = ov::as_type_ptr<snippets::op::Kernel>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(kernel != nullptr, "Invoked with invalid op argument");
    OV_CPU_JIT_EMITTER_ASSERT(!kernel->region->empty(), "Invoked with empty body");
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

//====================================================================================
// GPR    | Description                   | Usage             | Purpose
// ===================================================================================
// X0     | Argument register             | Use directly      | reg_runtime_params_idx
// X1     | Argument register             | Use directly      | Data pointer register
// X2     | Argument register             | Use directly      | Data pointer register
// X3     | Argument register             | Use directly      | Data pointer register
// X4     | Argument register             | Use directly      | Data pointer register
// X5     | Argument register             | Use directly      | Data pointer register
// X6     | Argument register             | Use directly      | Data pointer register
// X7     | Argument register             | Use directly      | Data pointer register
// X8     | Indirect result reg           | Use directly      | Data pointer register
// X9     | Caller-saved temp reg         | Use directly      | Data pointer register
// X10    | Caller-saved temp reg         | Use directly      | Data pointer register
// X11    | Caller-saved temp reg         | Use directly      | Data pointer register
// X12    | Caller-saved temp reg         | Use directly      | Data pointer register
// X13    | Caller-saved temp reg         | Use directly      | Data pointer register
// X14    | Caller-saved temp reg         | Use directly      | Data pointer register
// X15    | Caller-saved temp reg         | Use directly      | Data pointer register
// X16    | Intra-procedure-call temp reg | Saved in preamble | Data pointer register
// X17    | Intra-procedure-call temp reg | Saved in preamble | Data pointer register
// X18    | Platform register             | Do not use        | Do not use
// X19    | Callee-saved register         | Saved in preamble | Data pointer register
// X20    | Callee-saved register         | Saved in preamble | Data pointer register
// X21    | Callee-saved register         | Saved in preamble | Data pointer register
// X22    | Callee-saved register         | Saved in preamble | Data pointer register
// X23    | Callee-saved register         | Saved in preamble | X_TMP_0
// X24    | Callee-saved register         | Saved in preamble | X_TMP_1
// X25    | Callee-saved register         | Saved in preamble | Data pointer register
// X26    | Callee-saved register         | Saved in preamble | Data pointer register
// X27    | Callee-saved register         | Saved in preamble | Data pointer register
// X28    | Callee-saved register         | Saved in preamble | X_DEFAULT_ADDR
// X29    | Frame pointer register (FP)   | Saved in preamble | Frame pointer register
// X30    | Link register (LR)            | Saved in preamble | Data pointer register
// X31    | Stack Pointer (SP)            | Use directly      | Stack Pointer
//====================================================================================
// Note that 2 of the 25 marked Data pointer registers will be used as work_amounts in
// two-level loops, so the actual number of Data pointer register is 23.
//====================================================================================
void jit_kernel_emitter::init_reg_pools(const std::set<size_t>& gpr_blacklist, const std::set<size_t>& vec_blacklist) {
    gp_regs_pool.resize(32);
    vec_regs_pool.resize(32);
    // It's easier to remove the last item during mapping, so fill descending to map ascending
    for (size_t i = 0; i < 32; i++)
        gp_regs_pool[i] = vec_regs_pool[i] = 31 - i;
    auto remove_regs_from_pool = [](std::vector<size_t>& pool, const std::set<size_t>& to_remove) {
        // It's important to keep the order of other elements
        pool.erase(std::remove_if(pool.begin(), pool.end(),
                                  [&](size_t x) {return to_remove.count(x) != 0;}), pool.end());
    };
    std::set<size_t> gprs_blacklist_extended{Operand::X18, Operand::X23, Operand::X24, Operand::X28, Operand::X29, Operand::SP};
    gprs_blacklist_extended.insert(gpr_blacklist.begin(), gpr_blacklist.end());
    // Reserve reg_indexes_idx and reg_runtime_params_idx, since they'll be used to pass runtime call args to kernel
    remove_regs_from_pool(gp_regs_pool, gprs_blacklist_extended);
    remove_regs_from_pool(vec_regs_pool, vec_blacklist);
}

void jit_kernel_emitter::emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out,
                                   const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void jit_kernel_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.empty() && out.empty(), ": Expects 0 registers on input and output");
    const auto num_params = num_inputs + num_outputs + num_unique_buffers;
    // The number of used gpr may be >= num_params since LoopBegin+LoopEnd could also use gpr to store work_amount
    OV_CPU_JIT_EMITTER_ASSERT(data_ptr_regs_idx.size() == num_params,
                              "Number of inputs and outputs is inconsistent with the number of allocated registers ", num_params,
                              " data_ptr_regs_idx.size() = ", data_ptr_regs_idx.size());
}

void jit_kernel_emitter::init_body_regs(const std::set<size_t>& kernel_regs,
                                        const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) {
    // Initialize pools of gp and vec registers
    // Reserve kernel regs (reg_indexes_idx and, if there is, reg_runtime_params_idx), since they'll be used to pass runtime call args to kernel
    init_reg_pools(kernel_regs, {});

    mapping_info gpr_map_pool({}, gp_regs_pool);
    mapping_info vec_map_pool({}, vec_regs_pool);

    // Note that we can't use kernel_regs to store data pointers because
    // these regs are used to calculate offsets for the data pointers
    map_abstract_registers(gpr_map_pool, vec_map_pool, mem_access_exprs);
    for (const auto& abstract_to_physical : gpr_map_pool.first)
        data_ptr_regs_idx.push_back(abstract_to_physical.second);

    vec_map_pool.second.insert(vec_map_pool.second.end(), pool_vec_idxs.cbegin(), pool_vec_idxs.cend());
    gpr_map_pool.second.insert(gpr_map_pool.second.end(), pool_gpr_idxs.cbegin(), pool_gpr_idxs.cend());
    map_abstract_registers(gpr_map_pool, vec_map_pool, general_exprs);
}

void jit_kernel_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    h->preamble();

    auto data_ptr_regs = transform_idxs_to_regs(data_ptr_regs_idx);

    init_data_pointers(data_ptr_regs);
    for (const auto& expression : *body) {
        const auto reg_info = expression->get_reg_info();
        auto in_regs = transform_snippets_regs_to_idxs(reg_info.first);
        auto out_regs = transform_snippets_regs_to_idxs(reg_info.second);
        const auto& emitter = expression->get_emitter();
        emitter->emit_code(in_regs, out_regs, vec_regs_pool, gp_regs_pool);
    }

    h->postamble();
}

jit_kernel_static_emitter::jit_kernel_static_emitter(dnnl::impl::cpu::aarch64::jit_generator* h, dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                                                     const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_kernel_emitter(h, isa, expr), reg_indexes_idx(Operand::X1) {
    const auto kernel = ov::as_type_ptr<snippets::op::KernelStatic>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(kernel != nullptr, "Expectes KernelStatic expression");
    jcp = *reinterpret_cast<const jit_snippets_compile_args*>(kernel->compile_params);
    master_shape = jcp.exec_domain;
    data_offsets = jcp.data_offsets;
    OV_CPU_JIT_EMITTER_ASSERT(data_offsets.size() == num_inputs + num_outputs, "Incompatible count of data offsets!");
    OV_CPU_JIT_EMITTER_ASSERT(data_offsets.front().size() == master_shape.size(), "Incompatible rank of data offsets!");

    // - Reserve reg_indexes_idx and reg_runtime_params_idx, since they'll be used to pass runtime call args to kernel
    // - However we can use reg_indexes_idx for non memory access operations
    //   since we won't need them after offsets calculation
    init_body_regs({reg_indexes_idx, reg_runtime_params_idx}, {}, {reg_indexes_idx});
}

void jit_kernel_static_emitter::init_data_pointers(const std::vector<XReg>& data_ptr_regs) const {
    XReg reg_indexes = XReg(static_cast<int>(reg_indexes_idx));
    XReg reg_runtime_params = XReg(static_cast<int>(reg_runtime_params_idx));
    XReg reg_tmp = XReg(h->X_TMP_0);
    XReg reg_aux = XReg(h->X_TMP_1);

    const auto num_params = num_inputs + num_outputs;
    // Note that we don't need offset for the last dim, since it's handled directly by Tile emitter
    const size_t offset_rank = master_shape.size() - 1;

    // master_shape size must be valid in both static and dynamic cases
    auto init_ptr_with_offset = [&](XReg pointer, const std::vector<size_t>& offsets) {
        for (size_t j = 0; j < offset_rank; j++) {
            if (master_shape[j] != 1 && offsets[j] != 0) {
                h->mov(reg_tmp, offsets[j]);
                h->ldr(reg_aux, ptr(reg_indexes, static_cast<int32_t>(j * sizeof(size_t))));
                h->mul(reg_tmp, reg_tmp, reg_aux);
                h->add(pointer, pointer, reg_tmp);
            }
        }
    };
    // Vector "data_ptr_regs" is sorted by abstract regs.
    // It means that the vector contains the physical registers in order [src, .., src, dst, .., dst, buffer]
    // So we can initialize buffer register firstly as last value of vector "data_ptr_regs"
    // NOTE: Snippets Buffer Scratchpad has the common data pointer for all Buffers (even with different ID).
    //       The accessing memory is covered by correct offsets in each Buffer and the corresponding MemoryAccess ops
    for (size_t i = 0; i < num_unique_buffers; i++) {
        h->ldr(data_ptr_regs[num_params + i], ptr(reg_runtime_params, static_cast<int32_t>(GET_OFF(buffer_scratchpad_ptr))));
    }
    for (size_t i = 0; i < num_params; i++) {
        if (i < num_inputs)
            h->ldr(data_ptr_regs[i], ptr(reg_runtime_params, static_cast<int32_t>(GET_OFF(src_ptrs) + i * sizeof(void*))));
        else
            h->ldr(data_ptr_regs[i], ptr(reg_runtime_params, static_cast<int32_t>(GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*))));
        init_ptr_with_offset(data_ptr_regs[i], data_offsets[i]);
    }
}

jit_kernel_dynamic_emitter::jit_kernel_dynamic_emitter(dnnl::impl::cpu::aarch64::jit_generator* h, dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                                                       const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_kernel_emitter(h, isa, expr) {
    const auto kernel = ov::as_type_ptr<snippets::op::KernelDynamic>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(kernel, "Expectes KernelDynamic expression");

    // - Reserve reg_runtime_params_idx, since it wll be used to pass runtime call args to all dynamic emitters that needs runtime args
    // - We cannot assign this register to the body emitters since runtime params MUST be valid during whole execution
    //   for all dynamic emitters
    init_body_regs({reg_runtime_params_idx});
}

void jit_kernel_dynamic_emitter::init_data_pointers(const std::vector<XReg>& data_ptr_regs) const {
    XReg reg_runtime_params = XReg(static_cast<int>(reg_runtime_params_idx));

    const auto num_params = num_inputs + num_outputs;
    for (size_t i = 0; i < num_unique_buffers; ++i) {
        h->ldr(data_ptr_regs[num_params + i], ptr(reg_runtime_params, static_cast<int32_t>(GET_OFF(buffer_scratchpad_ptr))));
    }
    for (size_t i = 0; i < num_params; i++) {
        if (i < num_inputs)
            h->ldr(data_ptr_regs[i], ptr(reg_runtime_params, static_cast<int32_t>(GET_OFF(src_ptrs) + i * sizeof(void*))));
        else
            h->ldr(data_ptr_regs[i], ptr(reg_runtime_params, static_cast<int32_t>(GET_OFF(dst_ptrs) + (i - num_inputs) * sizeof(void*))));
    }
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
