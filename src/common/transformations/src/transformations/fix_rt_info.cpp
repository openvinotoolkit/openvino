// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/fix_rt_info.hpp"

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <vector>

#include "itt.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"

bool ov::pass::FixRtInfo::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(FixRtInfo);

    for (auto& node : f->get_ops()) {
        // Recursively apply transformation for sub-graph based operations
        if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::SubGraphOp>(node)) {
            if (auto sub_graph = sub_graph_node->get_function()) {
                run_on_model(sub_graph);
            }
        }
        
        auto& rt_info = node->get_rt_info();
        {
            auto it_info = rt_info.find("PrimitivesPriority");
            if (it_info != rt_info.end()) {
                if (it_info->second.is<ov::PrimitivesPriority>()) {
                    rt_info.emplace(ov::PrimitivesPriority::get_type_info_static(),
                                    it_info->second.as<ov::PrimitivesPriority>());
                }
                if (it_info->second.is<std::string>()) {
                    rt_info.emplace(ov::PrimitivesPriority::get_type_info_static(),
                                    ov::PrimitivesPriority{it_info->second.as<std::string>()});
                }
                rt_info.erase(it_info);
            }
        }
    }
    return false;
}
