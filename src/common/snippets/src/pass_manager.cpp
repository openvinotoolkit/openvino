// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass_manager.hpp"

namespace ov {
namespace snippets {
namespace pass {

Manager::PassPosition::pass_list_type::const_iterator
Manager::PassPosition::get_insert_position(const pass_list_type& pass_list) const {
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
} // namespace pass
}// namespace snippets
}// namespace ov
