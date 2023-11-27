// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/manager.hpp"
#include "openvino/pass/pass.hpp"
#include "openvino/pass/validate.hpp"

#include <typeinfo>


namespace ov {
namespace snippets {
namespace pass {
/**
* @brief PassPosition describes a particular position in a transformation pipeline,
    *       where a new transformation should be inserted.
    * @param pass_name name of the anchor pass, the new pass will be inserted before/after it.
    *        Empty pass_name could mean either beginning or the end of the pipeline depending on the `after` flag.
    *        No default value. Note that pass names namespaces are not supported, ov::PassName and snippets::PassName
    *        are considered identical.
    * @param after `true` if the new pass should be inserted before the anchor pass, `false` otherwise (default).
    *        If `pass_name` is empty, `true` means the end, and `false` - the beginning of the pipeline.
    * @param pass_instance the number of the pass with matching `pass_name` to be considered as the anchor pass.
    *        0 (default) means the first pass with `pass_name` will be considered as the anchor pass.
* @ingroup snippets
*/
class PassPosition {
public:
    enum class Place {Before, After, PipelineStart, PipelineEnd};
    explicit PassPosition(Place pass_place);
    explicit PassPosition(Place pass_place, std::string pass_name, size_t pass_instance = 0);

    template <typename PassBase>
    typename std::vector<std::shared_ptr<PassBase>>::const_iterator
    get_insert_position(const std::vector<std::shared_ptr<PassBase>>& pass_list) const {
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

        switch (m_place) {
        case Place::PipelineStart:
            return pass_list.cbegin();
        case Place::PipelineEnd:
            return pass_list.cend();
        case Place::Before:
        case Place::After: {
            auto insert_it = std::find_if(pass_list.cbegin(), pass_list.cend(), match);
            OPENVINO_ASSERT(insert_it != pass_list.cend(), "snippets::pass::Manager failed to find pass ", m_pass_name);
            return m_place == Place::After ? std::next(insert_it) : insert_it;
        }
        default:
            OPENVINO_THROW("Unsupported Place type in PassPosition::get_insert_position");
        }
    }

private:
    const std::string m_pass_name;
    const size_t m_pass_instance{0};
    const Place m_place{Place::Before};
};
} // namespace pass
} // namespace snippets
} // namespace ov
