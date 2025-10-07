// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "factory.hpp"

#include <algorithm>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_input.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/subtract.hpp"
#include "snippets/op/powerstatic.hpp"

namespace ov::intel_cpu::tpp::op {
namespace {
// Since we removed TPP eltwise classes, we also remove the custom power static builder
// that was creating TPP Reciprocal, Square, and SquareRoot nodes
}  // namespace

std::unordered_map<ov::DiscreteTypeInfo, NodeFactory::tpp_builder> NodeFactory::m_direct_mapping{
    // All TPP eltwise and reduce operations have been removed
};

std::vector<NodeFactory::TPPCustomBuilder> NodeFactory::m_custom_mapping{};

std::shared_ptr<ov::Node> NodeFactory::create(const std::shared_ptr<ov::Node>& n) {
    const auto& found = m_direct_mapping.find(n->get_type_info());
    std::shared_ptr<ov::Node> tpp_node{nullptr};
    if (found != m_direct_mapping.end()) {
        tpp_node = (found->second)(n);
    } else {
        for (const auto& custom_builder : m_custom_mapping) {
            if (custom_builder.matcher(n)) {
                tpp_node = custom_builder.builder(n);
                break;
            }
        }
    }
    return tpp_node;
}

bool NodeFactory::is_supported(const std::shared_ptr<ov::Node>& n) {
    auto matches = [=](const NodeFactory::TPPCustomBuilder& custom_builder) {
        return custom_builder.matcher(n);
    };
    // TPP currently supports only FP32 precisions (ticket: 130010)
    // Note: verify that TypeRelaxed property is maintained (mismatched input precisions)
    // after low precisions are enabled (ticket: 132328)
    const auto& ins = n->inputs();
    auto is_fp32_input = [](const ov::Input<ov::Node>& in) {
        return in.get_element_type() == element::f32;
    };
    const bool all_inputs_fp32 = std::all_of(ins.begin(), ins.end(), is_fp32_input);
    return ((m_direct_mapping.count(n->get_type_info()) != 0U) ||
            std::any_of(m_custom_mapping.begin(), m_custom_mapping.end(), matches)) &&
           all_inputs_fp32;
}

}  // namespace ov::intel_cpu::tpp::op
