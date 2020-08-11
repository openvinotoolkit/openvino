//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/rt_info.hpp"
#include "ngraph/node.hpp"
#include "ngraph/variant.hpp"

ngraph::Node::RTMap mergeRuntimeInfo(const ngraph::NodeVector& nodes)
{
    ngraph::Node::RTMap mergedInfo;
    for (auto& node : nodes)
    {
        for (auto& item : node->get_rt_info())
        {
            mergedInfo[item.first] = item.second;
        }
    }

    ngraph::Node::RTMap newInfo;
    for (auto& item : mergedInfo)
    {
        if (auto merge_attr = item.second->merge(nodes))
        {
            newInfo[item.second->get_type_info().name] = merge_attr;
        }
    }

    return newInfo;
}

void ngraph::copy_runtime_info(std::shared_ptr<ngraph::Node> from, std::shared_ptr<ngraph::Node> to)
{
    auto& rtInfoFrom = from->get_rt_info();
    auto& rtInfoTo = to->get_rt_info();
    rtInfoTo = rtInfoFrom;
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
