// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convert_to_unsigned_nms_gather.hpp"
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <queue>
#include <memory>
#include "itt.hpp"

using namespace ngraph;
using namespace std;

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertToUnsignedNmsGather, "ConvertToUnsignedNmsGather", 0);

void convert_gather_indices_to_unsigned(const ngraph::NodeVector& gather_nodes);

ngraph::NodeVector get_NmsGather_destinations(const std::shared_ptr<ngraph::Node>& nms_node);

bool ngraph::pass::ConvertToUnsignedNmsGather::run_on_function(std::shared_ptr<ngraph::Function> f) {
    MATCHER_SCOPE(MarkNMSGather);
    bool res = false;
    for (const auto& op : f->get_ordered_ops()) {
        // without dynamic_cast since not all NMS versions are inherited from the same base nms
        if (strcmp(op->get_type_name(), opset8::NonMaxSuppression::type_info.name) != 0)
            continue;
        convert_gather_indices_to_unsigned(get_NmsGather_destinations(op));
        res = true;
    }
    return res;
}

void convert_gather_indices_to_unsigned(const NodeVector& gather_nodes) {
    for (const auto& gather : gather_nodes) {
        auto indices = gather->input_value(1);

        auto out_type = element::Type_t::u32;
        if (indices.get_element_type() == element::Type_t::i64)
            out_type = element::Type_t::u64;

        if (auto old_convert = dynamic_pointer_cast<opset8::Convert>(indices.get_node_shared_ptr())) {
            old_convert->set_convert_element_type(out_type);
        } else {
            auto convert_to_unsigned = make_shared<opset8::Convert>(indices, out_type);
            gather->input(1).replace_source_output(convert_to_unsigned);
            copy_runtime_info(gather, convert_to_unsigned);
        }
    }
}

NodeVector get_NmsGather_destinations(const shared_ptr<Node>& nms_node) {
    NodeVector res;
    set<string> skip_node_types = {
            opset8::Squeeze::type_info.name,
            opset8::Unsqueeze::type_info.name,
            opset8::Reshape::type_info.name,
            opset8::Broadcast::type_info.name,
            opset8::StridedSlice::type_info.name,
            opset8::VariadicSplit::type_info.name,
            opset8::Concat::type_info.name,
            opset8::Convert::type_info.name
    };

    // BFS search
    std::unordered_set<shared_ptr<Node>> visited;
    std::queue<shared_ptr<Node>> stack;
    stack.push(nms_node->output(0).get_node_shared_ptr());

    while (!stack.empty()) {
        shared_ptr<Node> curr = stack.front();
        visited.insert(curr);
        stack.pop();

        for (const auto& next : curr->get_users()) {
            if (!visited.count(next) && skip_node_types.count(next->get_type_name())) {
                stack.push(next);
            } else if (!visited.count(next) && strcmp(next->get_type_name(), opset8::Gather::type_info.name) == 0) {
                // if next goes into Gather indices input
                if (next->input_value(1).get_node_shared_ptr() == curr) {
                    res.emplace_back(next);
                }
            }
        }
    }
    return res;
}
