// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/manager.hpp"


namespace ov {
namespace snippets {
namespace pass {

Manager::Manager(std::shared_ptr<ov::pass::PassConfig> pass_config, std::string name)
    : ov::pass::Manager(std::move(pass_config), std::move(name)) {}

std::shared_ptr<Manager::PassBase> Manager::register_pass_instance(const PassPosition& position,
                                                                   const std::shared_ptr<PassBase>& pass) {
    pass->set_pass_config(m_pass_config);
    return insert_pass_instance(position, pass);
}

void Manager::register_positioned_passes(const std::vector<PositionedPassBase>& pos_passes) {
    for (const auto& pp : pos_passes)
        register_pass_instance(pp.position, pp.pass);
}

std::shared_ptr<Manager::PassBase> Manager::insert_pass_instance(const PassPosition& position,
                                                                 const std::shared_ptr<PassBase>& pass) {
    auto insert_pos = position.get_insert_position(m_pass_list);
    insert_pos = m_pass_list.insert(insert_pos, pass);
    if (m_per_pass_validation) {
        // Note: insert_pos points to the newly inserted pass, so advance to validate the pass results
        std::advance(insert_pos, 1);
        m_pass_list.insert(insert_pos, std::make_shared<ov::pass::Validate>());
    }
    return pass;
}

} // namespace pass
} // namespace snippets
} // namespace ov
