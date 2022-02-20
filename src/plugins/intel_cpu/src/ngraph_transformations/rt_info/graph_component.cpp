// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/node.hpp>
#include "graph_component.hpp"

namespace ov {
namespace intel_cpu {
bool has_graph_component(const std::shared_ptr<ov::Node>& node) {
    return node->get_rt_info().count(GraphComponentAttr::get_type_info_static());
}

size_t get_graph_component(const std::shared_ptr<ov::Node>& node) {
    return node->get_rt_info().at(GraphComponentAttr::get_type_info_static()).as<GraphComponentAttr>().get_value();
}

void set_graph_component(const std::shared_ptr<ov::Node>& node, const size_t graph_component) {
    node->get_rt_info().emplace(GraphComponentAttr::get_type_info_static(), GraphComponentAttr{graph_component});
}
}  // namespace intel_cpu
}  // namespace ov