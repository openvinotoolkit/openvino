// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/init_registers.hpp"
#include "snippets/lowered/pass/init_live_ranges.hpp"
#include "snippets/lowered/pass/assign_registers.hpp"
#include "snippets/lowered/pass/insert_reg_spills.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

InitRegisters::InitRegisters(const std::shared_ptr<const Generator>& generator,
                             const std::shared_ptr<PassConfig>& pass_config) :
                             Pass(), m_reg_manager(generator), m_pass_config(pass_config) {
}

bool InitRegisters::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InitRegisters");
    lowered::pass::PassPipeline reg_pipeline(m_pass_config);
    reg_pipeline.register_pass<lowered::pass::InitLiveRanges>(m_reg_manager);
    reg_pipeline.register_pass<lowered::pass::AssignRegisters>(m_reg_manager);
    reg_pipeline.register_pass<lowered::pass::InsertRegSpills>(m_reg_manager);
    reg_pipeline.run(linear_ir);
    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

