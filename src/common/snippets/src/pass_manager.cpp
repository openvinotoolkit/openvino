// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass_manager.hpp"

namespace ov {
namespace snippets {
namespace pass {

Manager::PassPosition::PassPosition(std::string pass_name, bool after, size_t pass_instance)
: m_pass_name(std::move(pass_name)), m_pass_instance(pass_instance), m_after(after) {
    OPENVINO_ASSERT((!m_pass_name.empty()) || (m_pass_instance == 0),
                    "Non-zero pass_instance is not allowed if pass_name is not provided");
}

Manager::PassPosition::PassListType::const_iterator
Manager::PassPosition::get_insert_position(const PassListType& pass_list) const {
    if (m_pass_name.empty())
        return m_after ? pass_list.cend() : pass_list.cbegin();
    size_t pass_count = 0;
    auto match = [this, &pass_count](const std::shared_ptr<PassBase>& p) {
        auto name = p->get_name();
        // Note that MatcherPass and ModelPass currently have different naming policies:
        // - MatcherPass have names without namespaces, e.g. ConvertToSwishCPU
        // - Similar ModelPass name includes namespaces: ov::snippets::pass::ConvertToSwishCPU
        // So we have to remove everything before the last ':', and ':' itself
        if (name.size() > m_pass_name.size()) {
            const auto pos = name.find_last_of(':');
            if (pos == std::string::npos)
                return false;
            name = name.substr(pos + 1);
        }
        if (name == m_pass_name) {
            if (m_pass_instance == pass_count)
                return true;
            pass_count++;
        }
        return false;
    };
    auto insert_it = std::find_if(pass_list.cbegin(), pass_list.cend(), match);
    OPENVINO_ASSERT(insert_it != pass_list.cend(), "snippets::pass::Manager failed to find pass ", m_pass_name);
    if (m_after)
        std::advance(insert_it, 1);
    return insert_it;
}

std::shared_ptr<Manager::PassBase> Manager::register_pass_instance(const PassPosition& position,
                                                                   const std::shared_ptr<PassBase>& pass) {
    pass->set_pass_config(m_pass_config);
    return insert_pass_instance(position, pass);
}

void Manager::register_positioned_passes(const std::vector<PositionedPass>& pos_passes) {
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
}// namespace snippets
}// namespace ov
