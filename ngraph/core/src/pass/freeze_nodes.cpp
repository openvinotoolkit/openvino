// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/pass/freeze_nodes.hpp>
NGRAPH_RTTI_DEFINITION(ngraph::pass::FreezeNodes, "FreezeNodes", 0);

bool ngraph::pass::FreezeNodes::run_on_function(std::shared_ptr<ngraph::Function> f)
{
    NGRAPH_CHECK(m_nodes_to_freeze.size() == m_replacing_values.size(),
                 "FreezeNodes transformation failed. "
                 "The number of nodes to be replaced and "
                 "the number of provided values do not match.");
    bool was_applied = false;
    for (const auto& node : f->get_ops())
    {
        auto node_to_freeze = find(m_nodes_to_freeze.begin(), m_nodes_to_freeze.end(), node);
        if (node_to_freeze != m_nodes_to_freeze.end())
        {
            auto idx = std::distance(m_nodes_to_freeze.begin(), node_to_freeze);
            NGRAPH_CHECK(node->outputs().size() == m_replacing_values[idx].size(),
                         "FreezeNodes transformation failed."
                         "Values are not provided for all node outputs."
                         "Node",
                         node->get_friendly_name(),
                         " has ",
                         node->outputs().size(),
                         "outputs, but count of provided values is ",
                         m_replacing_values[idx].size());

            OutputVector replacing_consts;
            for (size_t i = 0; i < node->outputs().size(); ++i)
            {
                const auto& replacement_output = node->output(i);
                auto data = std::make_shared<HostTensor>(replacement_output.get_element_type(),
                                                         replacement_output.get_partial_shape(),
                                                         "");
                data->write(m_replacing_values[idx][i].data(), m_replacing_values[idx][i].size());
                auto replacing_const = std::make_shared<opset6::Constant>(data);
                replacing_consts.push_back(replacing_const);
            }
            replace_node(node, replacing_consts);
            if (const auto& param = std::dynamic_pointer_cast<opset6::Parameter>(node))
            {
                f->remove_parameter(param);
            }
            else if (const auto& sink = std::dynamic_pointer_cast<op::Sink>(node))
            {
                f->remove_sink(sink);
            }
            else if (const auto& res = std::dynamic_pointer_cast<opset6::Result>(node))
            {
                f->remove_result(res);
            }

            was_applied = true;
        }
    }

    return was_applied;
}

ngraph::pass::FreezeNodes::FreezeNodes(const ngraph::NodeVector& nodes_to_freeze,
                                       const std::vector<values_for_node>& replacing_values)
    : m_nodes_to_freeze(nodes_to_freeze)
    , m_replacing_values(replacing_values)
{
}
