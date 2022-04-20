// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/fix_rt_info.hpp"

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <vector>

#include "itt.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"

bool ngraph::pass::FixRtInfo::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    // TODO: enable conditional compile
    // RUN_ON_FUNCTION_SCOPE(FixRtInfo);

    for (auto& node : f->get_ops()) {
        // Recursively apply transformation for sub-graph based operations
        if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::SubGraphOp>(node)) {
            if (auto sub_graph = sub_graph_node->get_function()) {
                run_on_model(sub_graph);
            }
        }
        OPENVINO_SUPPRESS_DEPRECATED_START
        auto& rt_info = node->get_rt_info();
        {
            auto it_info = rt_info.find("PrimitivesPriority");
            if (it_info != rt_info.end()) {
                if (it_info->second.is<std::shared_ptr<ngraph::VariantWrapper<std::string>>>()) {
                    rt_info.emplace(
                        ov::PrimitivesPriority::get_type_info_static(),
                        ov::PrimitivesPriority{
                            it_info->second.as<std::shared_ptr<ngraph::VariantWrapper<std::string>>>()->get()});
                } else if (it_info->second.is<ov::PrimitivesPriority>()) {
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
        {
            auto it_info = rt_info.find("affinity");
            if (it_info != rt_info.end()) {
                if (it_info->second.is<std::shared_ptr<ngraph::VariantWrapper<std::string>>>()) {
                    it_info->second = it_info->second.as<std::shared_ptr<ngraph::VariantWrapper<std::string>>>()->get();
                }
            }
        }
        OPENVINO_SUPPRESS_DEPRECATED_END
    }
    return false;
}
