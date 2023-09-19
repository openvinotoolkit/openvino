// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/manager.hpp"

#include "snippets/pass/subgraph_pass.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/itt.hpp"

#include "ngraph/pass/pass.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "openvino/util/env_util.hpp"
#include "openvino/util/log.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace ov {
namespace snippets {
namespace pass {
Manager::PassPosition::PassPosition(Place pass_place) : m_place(pass_place) {
    OPENVINO_ASSERT(m_place == Place::PipelineStart || m_place == Place::PipelineEnd,
                    "Invalid arg: pass_name and pass_instance args could be omitted only for Place::PipelineStart/Place::PipelineEnd");
}
Manager::PassPosition::PassPosition(Place pass_place, std::string pass_name, size_t pass_instance)
: m_pass_name(std::move(pass_name)), m_pass_instance(pass_instance), m_place(pass_place) {
    OPENVINO_ASSERT((m_place == Place::Before || m_place == Place::After) && !m_pass_name.empty(),
                    "Invalid args combination: pass_place must be Place::Before/Place::After and pass_name must be non-empty");
}

Manager::PassPosition::PassListType::const_iterator
Manager::PassPosition::get_insert_position(const PassListType& pass_list) const {
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
        case Place::PipelineStart: return pass_list.cbegin();
        case Place::PipelineEnd: return pass_list.cend();
        case Place::Before:
        case Place::After: {
            auto insert_it = std::find_if(pass_list.cbegin(), pass_list.cend(), match);
            OPENVINO_ASSERT(insert_it != pass_list.cend(), "snippets::pass::Manager failed to find pass ", m_pass_name);
            return m_place == Place::After ?  std::next(insert_it) : insert_it;
        }
        default:
            OPENVINO_THROW("Unsupported Place type in PassPosition::get_insert_position");
    }
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

bool Manager::run_passes_on_subgraph(const std::shared_ptr<ov::snippets::op::Subgraph>& subgraph) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::Manager");

    OPENVINO_ASSERT(subgraph != nullptr, "ov::snippets::pass::Manager got invalid Subgraph op");

    static bool profile_enabled =
        ov::util::getenv_bool("NGRAPH_PROFILE_PASS_ENABLE") || ov::util::getenv_bool("OV_PROFILE_PASS_ENABLE");

    size_t index = 0;
    ngraph::stopwatch pass_timer;
    ngraph::stopwatch overall_timer;
    overall_timer.start();
    bool pass_applied = false;
    bool function_changed = false;
    bool needs_validate = false;
    const auto& body = subgraph->body_ptr();
    for (auto& pass : m_pass_list) {
        if (m_pass_config->is_disabled(pass->get_type_info())) {
            OPENVINO_DEBUG << "Pass " << pass->get_name() << " is disabled";
            continue;
        }

        OV_ITT_SCOPE(FIRST_INFERENCE, ov::pass::itt::domains::SnippetsTransform, ov::pass::perf_counters()[pass->get_type_info()]);

        pass_timer.start();

        if (auto subgraph_pass = std::dynamic_pointer_cast<ov::snippets::pass::SubgraphPass>(pass)) {
            // This checks is to skip the graph transformation when the graph pass relies on
            // static shape but the function state is dynamic.
            if (subgraph_pass->get_property(ov::pass::PassProperty::REQUIRE_STATIC_SHAPE) && body->is_dynamic()) {
                OPENVINO_DEBUG << "Pass " << pass->get_name() << " requires static shape but the "
                               << "model is dynamic. Skipping this transformation";
                continue;
            }

            pass_applied = subgraph_pass->run_on_subgraph(subgraph);
        } else if (auto matcher_pass = std::dynamic_pointer_cast<ov::pass::MatcherPass>(pass)) {
            // This checks is to skip the graph transformation when the graph pass relies on
            // static shape but the function state is dynamic.
            if (matcher_pass->get_property(ov::pass::PassProperty::REQUIRE_STATIC_SHAPE) && body->is_dynamic()) {
                OPENVINO_DEBUG << "Pass " << pass->get_name() << " requires static shape but the "
                               << "model is dynamic. Skipping this transformation";
                continue;
            }
            // GraphRewrite is a temporary container for MatcherPass to make execution
            // on on entire ov::Model
            pass_applied = ov::pass::GraphRewrite(matcher_pass).run_on_model(body);
        } else if (auto function_pass = std::dynamic_pointer_cast<ov::pass::ModelPass>(pass)) {
            // This checks is to skip the graph transformation when the graph pass relies on
            // static shape but the function state is dynamic.
            if (function_pass->get_property(ov::pass::PassProperty::REQUIRE_STATIC_SHAPE) && body->is_dynamic()) {
                OPENVINO_DEBUG << "Pass " << pass->get_name() << " requires static shape but the "
                               << "model is dynamic. Skipping this transformation";
                continue;
            }

            if (std::dynamic_pointer_cast<ov::pass::Validate>(pass)) {
                if (needs_validate) {
                    function_pass->run_on_model(body);
                    needs_validate = false;
                }
            } else {
                pass_applied = function_pass->run_on_model(body);
            }
        } else if (auto node_pass = std::dynamic_pointer_cast<ngraph::pass::NodePass>(pass)) {
            if (node_pass->get_property(ov::pass::PassProperty::REQUIRE_STATIC_SHAPE) && body->is_dynamic()) {
                OPENVINO_DEBUG << "Pass " << pass->get_name() << " requires static shape but the "
                               << "model is dynamic. Skipping this transformation";
                continue;
            }
            for (const std::shared_ptr<ov::Node>& n : body->get_ops()) {
                pass_applied |= node_pass->run_on_node(n);
            }
        }

        if (m_visualize) {
            // visualizations and serializations will be named after the outermost function
            const size_t num_digits_in_pass_index = 3;
            std::string index_str = std::to_string(index);
            index_str = std::string(num_digits_in_pass_index - index_str.length(), '0') + index_str;
            auto base_filename = subgraph->get_name() + std::string("_") + index_str + std::string("_") + pass->get_name();

            if (m_visualize) {
                auto file_ext = "svg";
                ov::pass::VisualizeTree vt(base_filename + std::string(".") + file_ext);
                vt.run_on_model(body);
            }
        }
        index++;
        pass_timer.stop();
        if (profile_enabled) {
            std::cout << std::setw(7) << pass_timer.get_milliseconds() << "ms " << pass->get_name() << "\n";
        }
        function_changed = function_changed || pass_applied;
        needs_validate = pass_applied;
    }
    if (profile_enabled) {
        std::cout << "passes done in " << overall_timer.get_milliseconds() << "ms\n";
    }
    OPENVINO_SUPPRESS_DEPRECATED_END

    return function_changed;
}

} // namespace pass
} // namespace snippets
} // namespace ov
