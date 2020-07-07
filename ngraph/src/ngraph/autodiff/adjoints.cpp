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

#include <list>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/strides.hpp"

using namespace ngraph;

Output<Node> make_broadcast_zero(const Output<Node>& output)
{
    Output<Node> zero = std::make_shared<op::ScalarConstantLike>(output, 0.0);
    Output<Node> bzero = std::make_shared<op::v0::BroadcastLike>(zero, output, AxisSet{});
    return bzero;
}

OutputVector make_zeros(std::shared_ptr<Node> x)
{
    OutputVector zeros;
    for (auto output : x->outputs())
    {
        zeros.push_back(make_broadcast_zero(output));
    }
    return zeros;
}

autodiff::Adjoints::Adjoints(const OutputVector& ys, const OutputVector& cs)
{
    if (ys.size() != cs.size())
    {
        throw ngraph_error("ys and cs must be equal size");
    }

    // Pass 1 determines which nodes contribute to y as well as setting up a reverse
    // topological sort.

    // Number of nodes that use the node's value
    std::unordered_map<std::shared_ptr<Node>, size_t> parent_counts;

    // Nodes we should check
    std::list<std::shared_ptr<Node>> nodes_to_check;
    for (auto& y : ys)
    {
        nodes_to_check.push_back(y.get_node_shared_ptr());
    }
    while (nodes_to_check.size() > 0)
    {
        auto node = nodes_to_check.front();
        nodes_to_check.pop_front();
        if (m_adjoint_map.find(node.get()) == m_adjoint_map.end())
        {
            m_adjoint_map[node.get()] = OutputVector(node->get_output_size());
            for (auto value : node->input_values())
            {
                auto arg = value.get_node_shared_ptr();
                auto count_it = parent_counts.find(arg);
                if (count_it == parent_counts.end())
                {
                    parent_counts[arg] = 1;
                    nodes_to_check.push_front(arg);
                }
                else
                {
                    parent_counts[arg]++;
                }
            }
        }
    }

    // Second pass visits the nodes so that all users of a node's value are visited
    // before a node is visited.
    for (size_t i = 0; i < ys.size(); i++)
    {
        add_delta(ys.at(i), cs.at(i));
    }

    for (auto& y : ys)
    {
        auto node = y.get_node_shared_ptr();
        if (find(nodes_to_check.begin(), nodes_to_check.end(), node) == nodes_to_check.end())
        {
            nodes_to_check.push_back(y.get_node_shared_ptr());
        }
    }

    while (nodes_to_check.size() > 0)
    {
        auto node = nodes_to_check.front();
        nodes_to_check.pop_front();
        // Look for nodes that will be available when this node is done
        for (auto value : node->input_values())
        {
            auto input_source_node = value.get_node_shared_ptr();
            auto count_it = parent_counts.find(input_source_node);
            count_it->second--;
            if (0 == count_it->second)
            {
                nodes_to_check.push_front(input_source_node);
            }
        }
        OutputVector deltas = m_adjoint_map[node.get()];
        for (size_t i = 0; i < node->get_output_size(); ++i)
        {
            auto& delta = deltas[i];
            if (delta == Output<Node>())
            {
                delta = make_broadcast_zero(node->output(i));
            }
        }
        node->generate_adjoints(*this, deltas);
    }
}

Output<Node> autodiff::Adjoints::backprop_output(const Output<Node>& x)
{
    auto node = x.get_node();
    auto adjoint_it = m_adjoint_map.find(node);
    Output<Node> result;
    OutputVector deltas;
    if (m_adjoint_map.end() == adjoint_it)
    {
        deltas = OutputVector(node->get_output_size());
        m_adjoint_map[node] = deltas;
    }
    else
    {
        deltas = adjoint_it->second;
    }
    if (deltas.at(x.get_index()) == Output<Node>())
    {
        deltas.at(x.get_index()) = make_broadcast_zero(x);
    }
    return deltas.at(x.get_index());
}

void autodiff::Adjoints::add_delta(const Output<Node>& x, const Output<Node>& delta)
{
    auto adjoint_it = m_adjoint_map.find(x.get_node());
    if (adjoint_it == m_adjoint_map.end())
    {
        m_adjoint_map[x.get_node()] = OutputVector(x.get_node()->get_output_size());
        adjoint_it = m_adjoint_map.find(x.get_node());
    }
    auto& deltas = adjoint_it->second[x.get_index()];
    if (deltas == Output<Node>())
    {
        deltas = delta;
    }
    else
    {
        deltas = std::make_shared<op::Add>(deltas, delta);
    }
}

// This doesn't need an index since slice can only sit on top of GOE
void autodiff::Adjoints::add_delta_to_slice(const Output<Node>& x,
                                            const Output<Node>& delta,
                                            const Coordinate& lower_bounds,
                                            const Coordinate& upper_bounds,
                                            const Strides& strides)
{
    if (!(x.get_element_type().compatible(delta.get_element_type())) ||
        !(x.get_partial_shape().rank().compatible(delta.get_partial_shape().rank())))
    {
        throw ngraph_error(
            "Autodiff internal error: Mismatch on backprop and op in add_delta_to_slice.");
    }

    auto adjoint_it = m_adjoint_map.find(x.get_node());
    auto& deltas = adjoint_it->second[x.get_index()];
    if (deltas == Output<Node>())
    {
        auto zero = make_broadcast_zero(x);
        deltas =
            std::make_shared<op::ReplaceSlice>(zero, delta, lower_bounds, upper_bounds, strides);
    }
    else
    {
        deltas = std::make_shared<op::ReplaceSlice>(
            deltas,
            std::make_shared<op::Add>(
                std::make_shared<op::Slice>(deltas, lower_bounds, upper_bounds, strides), delta),
            lower_bounds,
            upper_bounds,
            strides);
    }
}
