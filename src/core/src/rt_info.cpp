// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/rt_info.hpp"

#include "openvino/op/util/op_types.hpp"

namespace {

std::unordered_map<std::string, std::vector<ov::Any>> get_copyable_attrs(const ov::OutputVector& outputs,
                                                                         const ov::Output<ov::Node>& to) {
    std::unordered_map<std::string, std::vector<ov::Any>> attrs;
    for (const auto& output : outputs) {
        for (const auto& item : output.get_rt_info()) {
            bool copy = true;
            if (item.second.is<ov::RuntimeAttribute>()) {
                copy = item.second.as<ov::RuntimeAttribute>().is_copyable(to.get_node_shared_ptr());
            }
            if (copy) {
                attrs[item.first].push_back(item.second);
            }
        }
    }
    return attrs;
}

std::unordered_map<std::string, std::vector<ov::Any>> get_copyable_attrs(const ov::NodeVector& nodes,
                                                                         const std::shared_ptr<ov::Node>& to) {
    std::unordered_map<std::string, std::vector<ov::Any>> attrs;
    for (const auto& node : nodes) {
        for (const auto& item : node->get_rt_info()) {
            bool copy = item.first != "opset";
            if (item.second.is<ov::RuntimeAttribute>()) {
                copy = copy && item.second.as<ov::RuntimeAttribute>().is_copyable(to);
            }
            if (copy) {
                attrs[item.first].push_back(item.second);
            }
        }
    }
    return attrs;
}

template <typename T>
ov::Node::RTMap mergeRuntimeInfo(const std::vector<T>& items, const T& to) {
    std::unordered_map<std::string, std::vector<ov::Any>> attrs = get_copyable_attrs(items, to);

    ov::Node::RTMap merged_attrs;
    for (auto& item : attrs) {
        auto& attr = *item.second.begin();
        if (item.second.size() == 1) {
            merged_attrs[item.first] = std::move(attr);
        } else {
            if (attr.is<ov::RuntimeAttribute>()) {
                auto merge_attr = attr.as<ov::RuntimeAttribute>().merge(items);
                if (!merge_attr.empty()) {
                    merged_attrs[item.first] = std::move(merge_attr);
                }
            }
        }
    }

    return merged_attrs;
}

ov::Any get_opset(const ov::Node::RTMap& rt_info) {
    auto it = rt_info.find("opset");
    if (it != rt_info.end()) {
        return it->second;
    }
    return nullptr;
}

void assign_runtime_info(const ov::Node::RTMap& from, ov::Node::RTMap& to) {
    auto opset = get_opset(to);
    for (auto& item : from) {
        to[item.first] = item.second;
    }
    if (!opset.empty()) {
        to["opset"] = std::move(opset);
    }
}

ov::NodeVector list_with_constants(const ov::NodeVector& to) {
    ov::NodeVector ops = to;
    for (auto& node : to) {
        if (!node) {
            continue;
        }
        for (auto& input : node->inputs()) {
            auto source_node = input.get_source_output().get_node_shared_ptr();
            if (ov::op::util::is_constant(source_node) && (0 == source_node->get_rt_info().size())) {
                if (std::find(ops.begin(), ops.end(), source_node) == ops.end()) {
                    ops.push_back(source_node);
                }
            }
        }
    }
    return ops;
}

ov::OutputVector list_with_constants(const ov::OutputVector& to) {
    ov::OutputVector ops = to;
    for (auto& node : to) {
        for (auto& input : node.get_node()->inputs()) {
            auto source_node = input.get_source_output();
            if (ov::op::util::is_constant(source_node.get_node_shared_ptr()) &&
                (0 == source_node.get_rt_info().size())) {
                if (std::find(ops.begin(), ops.end(), source_node) == ops.end()) {
                    ops.push_back(source_node);
                }
            }
        }
    }
    return ops;
}
}  // namespace

void ov::copy_runtime_info(const std::shared_ptr<ov::Node>& from, const std::shared_ptr<ov::Node>& to) {
    return copy_runtime_info(ov::NodeVector{from}, ov::NodeVector{to});
}

void ov::copy_runtime_info(const std::shared_ptr<ov::Node>& from, ov::NodeVector to) {
    return copy_runtime_info(ov::NodeVector{from}, std::move(to));
}

void ov::copy_runtime_info(const ov::NodeVector& from, const std::shared_ptr<ov::Node>& to) {
    return copy_runtime_info(from, ov::NodeVector{to});
}

void ov::copy_runtime_info(const ov::NodeVector& from, ov::NodeVector to) {
    for (auto& node : list_with_constants(to)) {
        assign_runtime_info(mergeRuntimeInfo(from, node), node->get_rt_info());
    }
}

void ov::copy_output_runtime_info(const ov::OutputVector& from, ov::OutputVector to) {
    for (auto& node : list_with_constants(to)) {
        assign_runtime_info(mergeRuntimeInfo(from, node), node.get_rt_info());
    }
}
