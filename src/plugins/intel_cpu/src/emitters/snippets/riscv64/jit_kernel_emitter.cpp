// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_kernel_emitter.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "emitters/plugin/riscv64/jit_emitter.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "jit_snippets_emitters.hpp"
#include "nodes/kernels/riscv64/cpu_isa_traits.hpp"
#include "nodes/kernels/riscv64/jit_generator.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "snippets/emitter.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/kernel.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/op/reg_spill.hpp"
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
    OPENVINO_ASSERT(kernel != nullptr, "jit_kernel_emitter invoked with invalid op argument");
    OPENVINO_ASSERT(!kernel->region->empty(), "jit_kernel_emitter invoked with empty body");
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
                                        const std::vector<size_t>& pool_gpr_idxs) const {
    validate_arguments(in, out);
    aux_vec_idxs = pool_vec_idxs;
    aux_gpr_idxs = pool_gpr_idxs;
    emit_impl(in, out);
}

void jit_kernel_emitter::validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    OPENVINO_ASSERT(in.size() == get_inputs_num() && out.empty(), "Unexpected number of input/output arguments");
    const auto num_params = num_inputs + num_outputs + num_unique_buffers;
    // The number of used gpr may be >= num_params since LoopBegin+LoopEnd could also use gpr to store work_amount
    OPENVINO_ASSERT(data_ptr_regs_idx.size() == num_params,
                    "Number of inputs and outputs is inconsistent with the number of allocated registers ",
                    num_params,
                    " data_ptr_regs_idx.size() = ",
                    data_ptr_regs_idx.size());
}

void jit_kernel_emitter::emit_impl(const std::vector<size_t>& in,
                                   [[maybe_unused]] const std::vector<size_t>& out) const {
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
            OPENVINO_THROW("Unsupported emitter_in_out_map instance");
        }
    };
    // Provide up to two temporary GPRs for pointer initialization math
    std::vector<Xbyak_riscv::Reg> aux_tmp_regs{};
    if (!available_gpr.empty()) {
        auto it = available_gpr.begin();
        aux_tmp_regs.emplace_back(static_cast<int>(it->idx));
        ++it;
        if (it != available_gpr.end()) {
            aux_tmp_regs.emplace_back(static_cast<int>(it->idx));
        }
    }
    init_data_pointers(utils::transform_idxs_to_regs(in), data_ptr_regs, aux_tmp_regs);
    for (const auto& expression : *body) {
        const auto reg_info = expression->get_reg_info();
        const auto& emitter = std::dynamic_pointer_cast<jit_emitter>(expression->get_emitter());
        OPENVINO_ASSERT(emitter, "Unexpected emitter type");
        auto expected_in_type = snippets::RegType::undefined;
        auto expected_out_type = snippets::RegType::undefined;
        const auto& node = expression->get_node();
        // Note: A few operations are allowed to have mixed register types on their inputs (or outputs) => skip
        // validation here
        if (!ov::is_type_any_of<snippets::op::LoopEnd, snippets::op::LoopBegin, snippets::op::RegSpillBase>(node) &&
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
        emitter->snippets::Emitter::emit_code(in_regs, out_regs, vec_pool, gpr_pool);
    }

    h->postamble();
}

jit_kernel_static_emitter::jit_kernel_static_emitter(jit_generator_t* h,
                                                     cpu_isa_t isa,
                                                     const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_kernel_emitter(h, isa, expr) {
    const auto kernel = ov::as_type_ptr<snippets::op::KernelStatic>(expr->get_node());
    OPENVINO_ASSERT(kernel != nullptr, "jit_kernel_static_emitter expects KernelStatic expression");
    jcp = *reinterpret_cast<const jit_snippets_compile_args*>(kernel->compile_params);
    master_shape = jcp.exec_domain;
    data_offsets = jcp.data_offsets;
    OPENVINO_ASSERT(data_offsets.size() == num_inputs + num_outputs, "Incompatible count of data offsets!");
    OPENVINO_ASSERT(!data_offsets.empty() && data_offsets.front().size() == master_shape.size(),
                    "Incompatible rank of data offsets!");
}

void jit_kernel_static_emitter::init_data_pointers(const std::vector<Xbyak_riscv::Reg>& arg_regs,
                                                   const std::vector<Xbyak_riscv::Reg>& data_ptr_regs,
                                                   const std::vector<Xbyak_riscv::Reg>& aux_gprs) const {
    OPENVINO_ASSERT(arg_regs.size() == 2, "Invalid arg regs size");
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
    OPENVINO_ASSERT(ov::is_type<snippets::op::KernelDynamic>(expr->get_node()),
                    "jit_kernel_dynamic_emitter expects KernelDynamic expression");
}

void jit_kernel_dynamic_emitter::init_data_pointers(
    const std::vector<Xbyak_riscv::Reg>& arg_regs,
    const std::vector<Xbyak_riscv::Reg>& data_ptr_regs,
    [[maybe_unused]] const std::vector<Xbyak_riscv::Reg>& aux_gprs) const {
    OPENVINO_ASSERT(arg_regs.size() == 1, "Invalid arg regs size");
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
