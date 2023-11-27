// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/pass_pipeline.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

void PassPipeline::register_pass(const std::shared_ptr<Pass>& pass) {
    m_passes.push_back(pass);
}

void PassPipeline::run(LinearIR& linear_ir) const {
    for (const auto& pass : m_passes) {
        pass->run(linear_ir);
    }
}

void PassPipeline::register_positioned_passes(const std::vector<PositionedPass>& pos_passes) {
    for (const auto& pp : pos_passes)
        insert_pass_instance(pp.position, pp.pass);
}

void PassPipeline::insert_pass_instance(const ov::snippets::pass::PassPosition& position,
                                        const std::shared_ptr<Pass>& pass) {
    m_passes.insert(position.get_insert_position(m_passes), pass);
}

void SubgraphPassPipeline::register_pass(const std::shared_ptr<SubgraphPass>& pass) {
    m_passes.push_back(pass);
}

void SubgraphPassPipeline::run(const LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) const {
    for (const auto& pass : m_passes)
        pass->run(linear_ir, begin, end);
}

const std::vector<std::shared_ptr<SubgraphPass>>& SubgraphPassPipeline::get_passes() const {
    return m_passes;
}

void SubgraphPassPipeline::register_positioned_passes(const std::vector<PositionedPass>& pos_passes) {
    for (const auto& pp : pos_passes)
        insert_pass_instance(pp.position, pp.pass);
}

void SubgraphPassPipeline::insert_pass_instance(const ov::snippets::pass::PassPosition& position,
                                                const std::shared_ptr<SubgraphPass>& pass) {
    m_passes.insert(position.get_insert_position(m_passes), pass);
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
