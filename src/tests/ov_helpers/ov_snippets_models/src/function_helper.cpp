// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "function_helper.hpp"
#include "common_test_utils/data_utils.hpp"
#include <snippets/snippets_isa.hpp>
#include <snippets/op/subgraph.hpp>
#include "ov_models/builders.hpp"

namespace ov {
namespace test {
namespace snippets {

// TODO: workaround while element-wise operations after `Parameter` are not added in Subgraph
std::vector<std::shared_ptr<Node>> FunctionHelper::makePrerequisitesOriginal() {
    std::vector<std::shared_ptr<Node>> nodes;

    const auto parameter = std::make_shared<ov::opset1::Parameter>();
    parameter->set_friendly_name("parameter");
    nodes.push_back(parameter);

    const auto maxPool = std::make_shared<ov::opset1::MaxPool>(
        parameter,
        Strides{ 1, 1 }, // strides
        Shape{ 0, 0 },   // pads_begin
        Shape{ 0, 0 },   // pads_end
        Shape{ 1, 1 });  // kernel
    maxPool->set_friendly_name("maxPool");
    nodes.push_back(maxPool);

    return nodes;
}

std::shared_ptr<Node> FunctionHelper::applyPrerequisites(const std::shared_ptr<Node>& parent, const std::vector<std::shared_ptr<Node>>& prerequisites) {
    std::shared_ptr<ov::Node> currentParent;
    if (prerequisites.empty()) {
        currentParent = parent;
    } else {
        auto begin = prerequisites[0];
        if (is_type<ov::opset1::Parameter>(begin)) {
            begin = prerequisites[1];
        }
        begin->set_argument(0, parent);

        currentParent = *prerequisites.rbegin();
    }
    return currentParent;
}

std::shared_ptr<Node> FunctionHelper::getSubgraph(const std::shared_ptr<Model>& f, const int index) {
    int currentIndex = 0;
    std::shared_ptr<ov::snippets::op::Subgraph> subgraph;
    for (const auto& op : f->get_ordered_ops()) {
        auto tmp_subgraph = as_type_ptr<ov::snippets::op::Subgraph>(op);
        if (tmp_subgraph != nullptr) {
            if (index == currentIndex) {
                return tmp_subgraph;
            }
            subgraph = tmp_subgraph;
            currentIndex++;
        }
    }

    if (index != -1) {
        return nullptr;
    }
    return subgraph;
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
