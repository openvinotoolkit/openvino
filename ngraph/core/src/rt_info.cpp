// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/rt_info.hpp"
#include "ngraph/node.hpp"
#include "ngraph/variant.hpp"

ov::Node::RTMap mergeRuntimeInfo(const ov::NodeVector& nodes)
{
    std::unordered_map<std::string, std::vector<std::shared_ptr<ov::Variant>>> attrs;
    for (const auto& node : nodes)
    {
        for (const auto& item : node->get_rt_info())
        {
            if (item.second->is_copyable())
            {
                attrs[item.first].push_back(item.second);
            }
        }
    }

    ov::Node::RTMap merged_attrs;
    for (auto& item : attrs)
    {
        auto attr = *item.second.begin();
        if (item.second.size() == 1)
        {
            merged_attrs[item.first] = attr;
        }
        else if (auto merge_attr = attr->merge(nodes))
        {
            merged_attrs[item.first] = merge_attr;
        }
    }

    return merged_attrs;
}

void ov::copy_runtime_info(std::shared_ptr<ov::Node> from, std::shared_ptr<ov::Node> to)
{
    auto& attrs = to->get_rt_info();
    attrs.clear();

    for (const auto& item : from->get_rt_info())
    {
        if (item.second->is_copyable())
        {
            attrs[item.first] = item.second;
        }
    }
}

void ov::copy_runtime_info(std::shared_ptr<ov::Node> from, ov::NodeVector to)
{
    for (auto& op : to)
    {
        copy_runtime_info(from, op);
    }
}

void ov::copy_runtime_info(const ov::NodeVector& from, std::shared_ptr<ov::Node> to)
{
    auto& rtInfoTo = to->get_rt_info();
    rtInfoTo = mergeRuntimeInfo(from);
}

void ov::copy_runtime_info(const ov::NodeVector& from, ov::NodeVector to)
{
    auto mergedInfo = mergeRuntimeInfo(from);
    for (auto& node : to)
    {
        auto& rtInfoTo = node->get_rt_info();
        rtInfoTo = mergedInfo;
    }
}
