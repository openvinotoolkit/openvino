// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/pass.hpp"

#include "snippets/utils.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

PassPipeline::PassPipeline() : m_pass_config(std::make_shared<PassConfig>()) {}
PassPipeline::PassPipeline(const std::shared_ptr<PassConfig>& pass_config) : m_pass_config(pass_config) {
    OPENVINO_ASSERT(m_pass_config != nullptr, "PassConfig is not initialized!");
}

void PassPipeline::register_pass(const PassPosition& position, const std::shared_ptr<Pass>& pass) {
    m_passes.insert(position.get_insert_position(m_passes), pass);
}

void PassPipeline::register_pass(const std::shared_ptr<Pass>& pass) {
    m_passes.push_back(pass);
}

void PassPipeline::run(LinearIR& linear_ir) const {
    for (const auto& pass : m_passes) {
        if (m_pass_config->is_disabled(pass->get_type_info())) {
            continue;
        }
        pass->run(linear_ir);
    }
}

void PassPipeline::register_positioned_passes(const std::vector<PositionedPass>& pos_passes) {
    for (const auto& pp : pos_passes)
        register_pass(pp.position, pp.pass);
}

PassPipeline::PassPosition::PassPosition(Place pass_place)
    : m_pass_type_info(DiscreteTypeInfo()), m_pass_instance(0), m_place(pass_place) {
    OPENVINO_ASSERT(utils::one_of(m_place, Place::PipelineStart, Place::PipelineEnd),
                    "Invalid arg: pass_type_info and pass_instance args could be omitted only for Place::PipelineStart/Place::PipelineEnd");
}
PassPipeline::PassPosition::PassPosition(Place pass_place, const DiscreteTypeInfo& pass_type_info, size_t pass_instance)
    : m_pass_type_info(pass_type_info), m_pass_instance(pass_instance), m_place(pass_place) {
    OPENVINO_ASSERT(utils::one_of(m_place, Place::Before, Place::After) && m_pass_type_info != DiscreteTypeInfo(),
                    "Invalid args combination: pass_place must be Place::Before/Place::After and m_pass_type_info must be non-empty");
}

std::vector<std::shared_ptr<Pass>>::const_iterator
PassPipeline::PassPosition::get_insert_position(const std::vector<std::shared_ptr<Pass>>& pass_list) const {
    size_t pass_count = 0;
    auto match = [this, &pass_count](const std::shared_ptr<Pass>& p) {
        if (p->get_type_info() == m_pass_type_info) {
            if (m_pass_instance == pass_count)
                return true;
            pass_count++;
        }
        return false;
    };
    switch (m_place) {
        case Place::PipelineStart: return pass_list.cbegin();
        case Place::PipelineEnd: return pass_list.cend();
        case Place::Before:
        case Place::After: {
            auto insert_it = std::find_if(pass_list.cbegin(), pass_list.cend(), match);
            OPENVINO_ASSERT(insert_it != pass_list.cend(), "PassPipeline failed to find pass ", m_pass_type_info);
            return m_place == Place::After ? std::next(insert_it) : insert_it;
        }
        default:
            OPENVINO_THROW("Unsupported Place type in PassPosition::get_insert_position");
    }
}

PassPipeline::PositionedPass::PositionedPass(PassPosition arg_pos, std::shared_ptr<Pass> arg_pass)
    : position(std::move(arg_pos)), pass(std::move(arg_pass)) {}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
