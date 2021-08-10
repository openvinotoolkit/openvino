// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/rt_info.hpp"
#include "ngraph/node.hpp"
#include "ngraph/variant.hpp"

ngraph::Node::RTMap mergeRuntimeInfo(const ngraph::NodeVector& nodes)
{
    std::unordered_map<std::string, std::vector<std::shared_ptr<ngraph::Variant>>> attrs;
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

    ngraph::Node::RTMap merged_attrs;
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

void ngraph::copy_runtime_info(std::shared_ptr<ngraph::Node> from, std::shared_ptr<ngraph::Node> to)
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

void ngraph::copy_runtime_info(std::shared_ptr<ngraph::Node> from, ngraph::NodeVector to)
{
    for (auto& op : to)
    {
        copy_runtime_info(from, op);
    }
}

void ngraph::copy_runtime_info(const ngraph::NodeVector& from, std::shared_ptr<ngraph::Node> to)
{
    auto& rtInfoTo = to->get_rt_info();
    rtInfoTo = mergeRuntimeInfo(from);
}

void ngraph::copy_runtime_info(const ngraph::NodeVector& from, ngraph::NodeVector to)
{
    auto mergedInfo = mergeRuntimeInfo(from);
    for (auto& node : to)
    {
        auto& rtInfoTo = node->get_rt_info();
        rtInfoTo = mergedInfo;
    }
}
