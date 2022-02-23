// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/rt_info.hpp"

#include "ngraph/variant.hpp"

namespace {

std::unordered_map<std::string, std::vector<ov::Any>> get_copyable_attrs(const ov::OutputVector& outputs) {
    std::unordered_map<std::string, std::vector<ov::Any>> attrs;
    for (const auto& output : outputs) {
        for (const auto& item : output.get_rt_info()) {
            bool copy = true;
            if (item.second.is<ov::RuntimeAttribute>()) {
                copy = item.second.as<ov::RuntimeAttribute>().is_copyable();
            }
            if (copy) {
                attrs[item.first].push_back(item.second);
            }
        }
    }
    return attrs;
}

std::unordered_map<std::string, std::vector<ov::Any>> get_copyable_attrs(const ov::NodeVector& nodes) {
    std::unordered_map<std::string, std::vector<ov::Any>> attrs;
    for (const auto& node : nodes) {
        for (const auto& item : node->get_rt_info()) {
            bool copy = item.first != "opset";
            if (item.second.is<ov::RuntimeAttribute>()) {
                copy = copy && item.second.as<ov::RuntimeAttribute>().is_copyable();
            }
            if (copy) {
                attrs[item.first].push_back(item.second);
            }
        }
    }
    return attrs;
}

template <typename T>
ov::Node::RTMap mergeRuntimeInfo(const T& items) {
    std::unordered_map<std::string, std::vector<ov::Any>> attrs = get_copyable_attrs(items);

    ov::Node::RTMap merged_attrs;
    for (auto& item : attrs) {
        auto attr = *item.second.begin();
        if (item.second.size() == 1) {
            merged_attrs[item.first] = attr;
        } else {
            if (attr.is<ov::RuntimeAttribute>()) {
                auto merge_attr = attr.as<ov::RuntimeAttribute>().merge(items);
                if (!merge_attr.empty()) {
                    merged_attrs[item.first] = merge_attr;
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
        to["opset"] = opset;
    }
}

}  // namespace

void ov::copy_runtime_info(const std::shared_ptr<ov::Node>& from, const std::shared_ptr<ov::Node>& to) {
    auto& attrs = to->get_rt_info();
    auto opset = get_opset(attrs);

    for (const auto& item : from->get_rt_info()) {
        bool copy = item.first != "opset";
        if (item.second.is<ov::RuntimeAttribute>()) {
            copy = copy && item.second.as<ov::RuntimeAttribute>().is_copyable();
        }
        if (copy) {
            attrs[item.first] = item.second;
        }
    }

    if (!opset.empty()) {
        attrs["opset"] = opset;
    }
}

void ov::copy_runtime_info(const std::shared_ptr<ov::Node>& from, ov::NodeVector to) {
    for (auto& op : to) {
        copy_runtime_info(from, op);
    }
}

void ov::copy_runtime_info(const ov::NodeVector& from, const std::shared_ptr<ov::Node>& to) {
    auto& rtInfoTo = to->get_rt_info();
    assign_runtime_info(mergeRuntimeInfo(from), rtInfoTo);
}

void ov::copy_runtime_info(const ov::NodeVector& from, ov::NodeVector to) {
    auto mergedInfo = mergeRuntimeInfo(from);
    for (auto& node : to) {
        auto& rtInfoTo = node->get_rt_info();
        assign_runtime_info(mergedInfo, rtInfoTo);
    }
}

void ov::copy_output_runtime_info(const ov::OutputVector& from, ov::OutputVector to) {
    auto mergedInfo = mergeRuntimeInfo(from);
    for (auto& node : to) {
        auto& rtInfoTo = node.get_rt_info();
        assign_runtime_info(mergedInfo, rtInfoTo);
    }
}
