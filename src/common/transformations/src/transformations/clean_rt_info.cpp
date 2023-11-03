// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/clean_rt_info.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/op/util/sub_graph_base.hpp"

namespace {
class TestVisitor : public ov::AttributeVisitor {
public:
    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {}
};

bool can_erase_key(const ov::Any& obj) {
    if (!obj.is<ov::RuntimeAttribute>())
        return true;
    const ov::RuntimeAttribute& casted = obj.as<const ov::RuntimeAttribute&>();
    TestVisitor visitor;
    return !casted.visit_attributes(visitor);
}

void clear_rt_info(ov::RTMap& rtInfo) {
    for (auto it = rtInfo.cbegin(); it != rtInfo.cend();) {
        if (can_erase_key(it->second)) {
            it = rtInfo.erase(it);
        } else {
            ++it;
        }
    }
}
}  // namespace

bool ov::pass::CleanRtInfo::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(CleanRtInfo);

    for (auto& node : f->get_ops()) {
        // Recursively apply transformation for sub-graph based operations
        if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::SubGraphOp>(node)) {
            if (auto sub_graph = sub_graph_node->get_function()) {
                run_on_model(sub_graph);
            }
        }
        auto& rtInfo = node->get_rt_info();
        clear_rt_info(rtInfo);
    }
    return false;
}
