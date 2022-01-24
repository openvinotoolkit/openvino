// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/rt_info.hpp"

#include "ngraph/node.hpp"
#include "ngraph/variant.hpp"

namespace {

ngraph::Node::RTMap mergeRuntimeInfo(const ngraph::NodeVector& nodes) {
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

    ngraph::Node::RTMap merged_attrs;
    for (auto& item : attrs) {
        auto attr = *item.second.begin();
        if (item.second.size() == 1) {
            merged_attrs[item.first] = attr;
        } else {
            if (attr.is<ov::RuntimeAttribute>()) {
                auto merge_attr = attr.as<ov::RuntimeAttribute>().merge(nodes);
                if (!merge_attr.empty()) {
                    merged_attrs[item.first] = merge_attr;
                }
            }
        }
    }

    return merged_attrs;
}

ov::Any get_opset(const ngraph::Node::RTMap& rt_info) {
    auto it = rt_info.find("opset");
    if (it != rt_info.end()) {
        return it->second;
    }
    return nullptr;
}

void assign_runtime_info(const ngraph::Node::RTMap& from, ngraph::Node::RTMap& to) {
    auto opset = get_opset(to);
    to = from;
    if (!opset.empty()) {
        to["opset"] = opset;
    }
}

}  // namespace

void ngraph::copy_runtime_info(std::shared_ptr<ngraph::Node> from, std::shared_ptr<ngraph::Node> to) {
    auto& attrs = to->get_rt_info();
    auto opset = get_opset(attrs);
    attrs.clear();

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

void ngraph::copy_runtime_info(std::shared_ptr<ngraph::Node> from, ngraph::NodeVector to) {
    for (auto& op : to) {
        copy_runtime_info(from, op);
    }
}

void ngraph::copy_runtime_info(const ngraph::NodeVector& from, std::shared_ptr<ngraph::Node> to) {
    auto& rtInfoTo = to->get_rt_info();
    assign_runtime_info(mergeRuntimeInfo(from), rtInfoTo);
}

void ngraph::copy_runtime_info(const ngraph::NodeVector& from, ngraph::NodeVector to) {
    auto mergedInfo = mergeRuntimeInfo(from);
    for (auto& node : to) {
        auto& rtInfoTo = node->get_rt_info();
        assign_runtime_info(mergedInfo, rtInfoTo);
    }
}
