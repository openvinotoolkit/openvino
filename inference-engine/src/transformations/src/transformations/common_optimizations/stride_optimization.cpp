// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include <transformations/common_optimizations/stride_optimization.hpp>
#include <ngraph/opsets/opset7.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::StrideOptimization, "StrideOptimization", 0);

bool ngraph::pass::StrideOptimization::run_on_function(std::shared_ptr<ngraph::Function> f) {
    RUN_ON_FUNCTION_SCOPE(StrideOptimization);
    bool rewritten = false;
    auto nodes = f->get_ordered_ops();
    for (auto it = nodes.rbegin(); it != nodes.rend(); it++) {
        auto& node = *it;
        rewritten |= handle_node(node);
        // Recursively apply transformation for sub-graph based operations
        if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::SubGraphOp>(node)) {
            if (auto sub_graph = sub_graph_node->get_function()) {
                rewritten |= run_on_function(sub_graph);
            }
        }
    }
    return rewritten;
}

bool ngraph::pass::StrideOptimization::handle_node(std::shared_ptr<ngraph::Node>& node) {
    const auto& info = node->get_type_info();
    if (info == opset7::Convolution::type_info) {
        return conv_stride_propagation(node);
    } else if (info == opset7::Relu::type_info ||
               info == opset7::Maximum::type_info ||
               info == opset7::Add::type_info ||
               info == opset7::Multiply::type_info) {
        return simple_stride_propagation(node, true);
    } else if (info == opset7::Constant::type_info ||
               info == opset7::Result::type_info) {
        return false;
    }
    return simple_stride_propagation(node, false);
}

static bool check_convolution(const std::shared_ptr<ngraph::Node>& conv) {
    const auto& kernel_pshape = conv->input_value(1).get_partial_shape();
    if (kernel_pshape.is_dynamic())
        return false;
    const auto& kernel_shape = kernel_pshape.get_shape();
    return std::all_of(kernel_shape.begin() + 2, kernel_shape.end(), [] (size_t s) -> bool { return s == 1; });
}

bool ngraph::pass::StrideOptimization::conv_stride_propagation(std::shared_ptr<ngraph::Node>& node) {
    auto conv = std::dynamic_pointer_cast<opset7::Convolution>(node);
    if (!conv)
        return false;
    const auto& conv_strides = conv->get_strides();
    const auto& next_ops = conv->get_users();
    bool all_ops_are_valid;
    std::vector<Strides> strides_vec;
    std::tie(strides_vec, all_ops_are_valid) = check_next_ops(next_ops);

    if (!all_ops_are_valid) {
        for (const auto& op : next_ops) {
            auto it = m_strides_map.find(op->get_friendly_name());
            if (it == m_strides_map.end())
                continue;
            const auto& strides = it->second;
            if (!std::all_of(strides.begin(), strides.end(), [] (size_t s) -> bool { return s == 1; }))
                insert_pooling(conv, op, strides);
        }
    } else if (strides_vec.size() > 0) {
        auto new_strides = conv_strides;
        std::transform(new_strides.begin(), new_strides.end(), strides_vec[0].begin(), new_strides.begin(),
                [] (size_t s1, size_t s2) -> size_t { return s1 * s2; });
        conv->set_strides(new_strides);
        // set strides [1, 1, ..]  for next_ops
        Strides strides(new_strides.size(), 1);
        for (const auto& op : next_ops) {
            auto casted = std::dynamic_pointer_cast<opset7::Convolution>(op);
            if (casted)
                casted->set_strides(strides);
        }
    }

    if (check_convolution(conv)) {
        m_strides_map.insert({conv->get_friendly_name(), conv->get_strides()});
    } else {
        Strides strides(conv_strides.size(), 1);
        m_strides_map.insert({conv->get_friendly_name(), strides});
    }

    return true;
}

bool ngraph::pass::StrideOptimization::simple_stride_propagation(std::shared_ptr<ngraph::Node>& node, bool supported) {
    const auto& rank = node->get_output_partial_shape(0).rank();
    if (rank.is_dynamic() || rank.get_length() < 3)
        return false;

    const auto& next_ops = node->get_users();
    bool all_ops_are_valid;
    std::vector<Strides> strides_vec;
    std::tie(strides_vec, all_ops_are_valid) = check_next_ops(next_ops);
    Strides strides_ones(static_cast<size_t>(rank.get_length()) - 2, 1);

    if (!all_ops_are_valid || !supported) {
        for (const auto& op : next_ops) {
            auto it = m_strides_map.find(op->get_friendly_name());
            if (it == m_strides_map.end())
                continue;
            const auto& strides = it->second;
            bool are_strides_ones = std::all_of(strides.begin(), strides.end(),
                    [] (size_t s) -> bool { return s == 1; });
            bool is_conv = std::dynamic_pointer_cast<opset7::Convolution>(op) != nullptr;
            if (!are_strides_ones && !is_conv)
                insert_pooling(node, op, strides);
        }
        m_strides_map.insert({node->get_friendly_name(), strides_ones});
        return true;
    }

    for (const auto& op : next_ops) {
        auto casted = std::dynamic_pointer_cast<opset7::Convolution>(op);
        if (casted) {
            casted->set_strides(strides_ones);
        }
    }

    if (strides_vec.size() > 0) {
        m_strides_map.insert({node->get_friendly_name(), strides_vec[0]});
    } else {
        m_strides_map.insert({node->get_friendly_name(), strides_ones});
    }

    return true;
}

std::tuple<std::vector<ngraph::Strides>, bool> ngraph::pass::StrideOptimization::check_next_ops(const std::vector<std::shared_ptr<Node>>& next_ops) {
    std::vector<Strides> strides;
    for (const auto& op : next_ops) {
        auto it = m_strides_map.find(op->get_friendly_name());
        if (it != m_strides_map.end()) {
            strides.push_back(it->second);
        }
    }
    bool all_ops_are_valid = !(next_ops.size() != strides.size() || (strides.size() > 0 &&
                                                                     !std::all_of(strides.begin(), strides.end(), [&strides] (const Strides& s) -> bool {
                                                                         bool all_ones = std::all_of(s.begin(), s.end(),
                                                                                                     [] (size_t i) -> bool { return i == 1; });
                                                                         return s == strides[0] && !all_ones;
                                                                     })));
    return std::make_tuple(strides, all_ops_are_valid);
}

void ngraph::pass::StrideOptimization::insert_pooling(const std::shared_ptr<Node>& first, const std::shared_ptr<Node>& second, const Strides& strides) {
    auto pool = std::make_shared<opset7::MaxPool>(first, strides, Shape{}, Shape{}, Shape(strides.size(), 1));
    auto second_inputs = second->inputs();
    for (size_t i = 0; i < second_inputs.size(); i++) {
        if (second_inputs[i].get_source_output() == first) {
            second->set_argument(i, pool);
            break;
        }
    }
}
