// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "emitters/utils.hpp"
#include "openvino/core/type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/op/reg_spill.hpp"
#include "snippets/utils/reg_utils.hpp"

namespace ov::intel_cpu::kernel_emitter_utils {

inline std::set<ov::snippets::Reg> make_register_pool(const std::vector<size_t>& register_idxs,
                                                      ov::snippets::RegType register_type) {
    std::set<ov::snippets::Reg> pool;
    for (const auto idx : register_idxs) {
        pool.insert({register_type, idx});
    }
    return pool;
}

struct RegisterPools {
    std::set<ov::snippets::Reg> gpr;
    std::set<ov::snippets::Reg> vec;
};

inline RegisterPools make_register_pools(const std::vector<size_t>& aux_gpr_idxs,
                                         const std::vector<size_t>& aux_vec_idxs,
                                         const std::vector<size_t>& data_ptr_regs_idx) {
    auto available_gpr = make_register_pool(aux_gpr_idxs, ov::snippets::RegType::gpr);
    auto available_vec = make_register_pool(aux_vec_idxs, ov::snippets::RegType::vec);

    // Data pointers are globally live, so do not keep them in the GPR pool.
    for (const auto idx : data_ptr_regs_idx) {
        available_gpr.erase({ov::snippets::RegType::gpr, idx});
    }

    return {std::move(available_gpr), std::move(available_vec)};
}

template <typename Emitter>
std::pair<ov::snippets::RegType, ov::snippets::RegType> get_expected_reg_types(
    const std::shared_ptr<Emitter>& emitter) {
    using in_out_map_t = decltype(emitter->get_in_out_type());
    switch (emitter->get_in_out_type()) {
    case in_out_map_t::gpr_to_vec:
        return {ov::snippets::RegType::gpr, ov::snippets::RegType::vec};
    case in_out_map_t::gpr_to_gpr:
        return {ov::snippets::RegType::gpr, ov::snippets::RegType::gpr};
    case in_out_map_t::vec_to_gpr:
        return {ov::snippets::RegType::vec, ov::snippets::RegType::gpr};
    case in_out_map_t::vec_to_vec:
        return {ov::snippets::RegType::vec, ov::snippets::RegType::vec};
    default:
        OPENVINO_THROW("Unsupported emitter_in_out_map instance");
    }
}

template <typename Emitter, typename NoopEmitter>
void emit_kernel_body(const ov::snippets::lowered::LinearIR& body, const RegisterPools& register_pools) {
    for (const auto& expression : body) {
        const auto reg_info = expression->get_reg_info();
        const auto emitter = std::dynamic_pointer_cast<Emitter>(expression->get_emitter());
        OV_CPU_JIT_EMITTER_ASSERT(emitter, "Unexpected emitter type");

        auto expected_in_type = ov::snippets::RegType::undefined;
        auto expected_out_type = ov::snippets::RegType::undefined;
        const auto& node = expression->get_node();
        // A few operations can have mixed register types, so skip validation for them.
        if (!ov::is_type_any_of<ov::snippets::op::LoopEnd, ov::snippets::op::RegSpillBase>(node) &&
            !std::dynamic_pointer_cast<NoopEmitter>(emitter)) {
            std::tie(expected_in_type, expected_out_type) = get_expected_reg_types(emitter);
        }

        // The live set contains input registers. Also exclude the expression outputs from the pool.
        auto live_regs = expression->get_live_regs();
        for (const auto reg : reg_info.second) {
            live_regs.insert(reg);
        }

        std::vector<ov::snippets::Reg> pool_gpr;
        std::vector<ov::snippets::Reg> pool_vec;
        std::set_difference(register_pools.gpr.begin(),
                            register_pools.gpr.end(),
                            live_regs.begin(),
                            live_regs.end(),
                            std::back_inserter(pool_gpr));
        std::set_difference(register_pools.vec.begin(),
                            register_pools.vec.end(),
                            live_regs.begin(),
                            live_regs.end(),
                            std::back_inserter(pool_vec));

        const auto in_regs = ov::snippets::utils::transform_snippets_regs_to_idxs(reg_info.first, expected_in_type);
        const auto out_regs = ov::snippets::utils::transform_snippets_regs_to_idxs(reg_info.second, expected_out_type);
        const auto gpr_pool = ov::snippets::utils::transform_snippets_regs_to_idxs(pool_gpr);
        const auto vec_pool = ov::snippets::utils::transform_snippets_regs_to_idxs(pool_vec);
        emitter->emit_code(in_regs, out_regs, vec_pool, gpr_pool);
    }
}

}  // namespace ov::intel_cpu::kernel_emitter_utils
