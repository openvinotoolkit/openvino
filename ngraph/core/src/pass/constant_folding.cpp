//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include "ngraph/pass/constant_folding.hpp"
#include <ngraph/op/constant.hpp>
#include "ngraph/op/util/sub_graph_base.hpp"
#include "ngraph/rt_info.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConstantFolding, "ConstantFolding", 0);

bool ngraph::pass::ConstantFolding::run_on_function(std::shared_ptr<ngraph::Function> f)
{
    bool rewritten = pre_calculated_values_folding(f);

    for (const auto& node : f->get_ordered_ops())
    {
        if (rewritten)
        {
            node->validate_and_infer_types();
        }

        OutputVector replacements(node->get_output_size());
        if (node->constant_fold(replacements, node->input_values()))
        {
            NGRAPH_CHECK(replacements.size() == node->get_output_size(),
                         "constant_fold_default returned incorrect number of replacements for ",
                         node);

            for (size_t i = 0; i < replacements.size(); ++i)
            {
                auto node_output = node->output(i);
                auto replacement = replacements.at(i);
                if (replacement.get_node_shared_ptr() && (node_output != replacement))
                {
                    if (replacements.size() == 1)
                    {
                        replacement.get_node_shared_ptr()->set_friendly_name(
                            node->get_friendly_name());
                    }
                    else
                    {
                        replacement.get_node_shared_ptr()->set_friendly_name(
                            node->get_friendly_name() + "." + std::to_string(i));
                    }
                    node_output.replace(replacement);
                    // Propagate runtime info attributes to replacement consumer nodes
                    copy_runtime_info_to_target_inputs(node, replacement);

                    rewritten = true;
                }
            }
        }
        else
        {
            // recursively constant fold operators containing subgraphs (ie: TensorIterator, Loop)
            if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::SubGraphOp>(node))
            {
                if (const auto& sub_graph = sub_graph_node->get_function())
                {
                    rewritten |= run_on_function(sub_graph);
                }
            }
        }
    }

    return rewritten;
}

void ngraph::pass::ConstantFolding::copy_runtime_info_to_target_inputs(
    const std::shared_ptr<Node>& node, const Output<Node>& replacement)
{
    for (auto& input : replacement.get_target_inputs())
    {
        auto consumer = input.get_node()->shared_from_this();
        copy_runtime_info({node, consumer}, consumer);
    }
}

bool ngraph::pass::ConstantFolding::pre_calculated_values_folding(
    const std::shared_ptr<ngraph::Function>& f)
{
    deque<shared_ptr<Node>> nodes;
    set<shared_ptr<Node>> visited;
    for (auto& r : f->get_results())
        nodes.push_back(r);
    for (auto& r : f->get_sinks())
        nodes.emplace_back(r);

    bool rewritten = false;
    while (!nodes.empty())
    {
        auto curr_node = nodes.front();
        nodes.pop_front();
        if (visited.count(curr_node) || is_type<op::Constant>(curr_node))
            continue;
        visited.insert(curr_node);

        for (auto& input_value : curr_node->input_values())
        {
            if (input_value.get_tensor().has_and_set_bound())
            {
                auto input_node = input_value.get_node_shared_ptr();
                auto replacement =
                    std::make_shared<op::Constant>(input_value.get_tensor().get_lower_value());
                if (replacement && !is_type<op::Constant>(input_node))
                {
                    if (input_node->get_output_size() == 1)
                    {
                        replacement->set_friendly_name(input_node->get_friendly_name());
                    }
                    else
                    {
                        replacement->set_friendly_name(input_node->get_friendly_name() + "." +
                                                       std::to_string(input_value.get_index()));
                    }
                    input_value.replace(replacement);
                    // Propagate runtime info attributes to replacement consumer nodes
                    copy_runtime_info_to_target_inputs(input_node, replacement);

                    rewritten = true;
                }
            }
            else
            {
                // continue searching
                const auto& input_node = input_value.get_node_shared_ptr();
                nodes.push_front(input_node);
            }
        }
    }
    return rewritten;
}
