// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_container_emitter.hpp"
#include "emitters/utils.hpp"
#include "utils/general_utils.h"

namespace ov {
namespace intel_cpu {

void jit_container_emitter::map_abstract_registers(mapping_info& gpr_map_pool, mapping_info& vec_map_pool,
                                                   snippets::lowered::LinearIR::container& expressions) const {
    OV_CPU_JIT_EMITTER_ASSERT(!expressions.empty(), "Cannot map registers when there is no allocated_emitters provided");

    auto map_regs = [&](const std::vector<snippets::Reg>& abstract_regs) {
        std::vector<snippets::Reg> physical_regs = abstract_regs;
        for (size_t i = 0; i < abstract_regs.size(); ++i) {
            const auto& abstract_reg = abstract_regs[i];
            const auto& type = abstract_reg.type;
            const auto& abstract = abstract_reg.idx;
            OV_CPU_JIT_EMITTER_ASSERT(one_of(type, snippets::RegType::gpr, snippets::RegType::vec), "Incorrect reg type detected!");
            auto& mapping = type == snippets::RegType::gpr ? gpr_map_pool : vec_map_pool;
            auto& abstract_to_physical = mapping.first;
            auto& regs_pool = mapping.second;
            auto& physical = physical_regs[i];
            if (abstract_to_physical.count(abstract) == 0) {
                OV_CPU_JIT_EMITTER_ASSERT(!regs_pool.empty(), "Cannot map registers for jit_container_emitter: not enough regs in the pool");
                physical.idx = regs_pool.back();
                regs_pool.pop_back();
                abstract_to_physical[abstract] = physical.idx;
            } else {
                physical.idx = abstract_to_physical[abstract];
            }
        }
        return physical_regs;
    };

    for (const auto& expression : expressions) {
        std::vector<snippets::Reg> in_physical_regs, out_physical_regs;
        std::vector<snippets::Reg> in_abstract_regs, out_abstract_regs;
        std::tie(in_abstract_regs, out_abstract_regs) = expression->get_reg_info();
        in_physical_regs = map_regs(in_abstract_regs);
        out_physical_regs = map_regs(out_abstract_regs);
        expression->set_reg_info({in_physical_regs, out_physical_regs});
        if (auto container = std::dynamic_pointer_cast<jit_container_emitter>(expression->get_emitter()))
            container->map_abstract_registers(gpr_map_pool, vec_map_pool, expressions);
    }
}

}   // namespace intel_cpu
}   // namespace ov
