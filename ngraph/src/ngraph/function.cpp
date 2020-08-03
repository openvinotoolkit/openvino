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

#include <algorithm>
#include <list>
#include <memory>

#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

constexpr DiscreteTypeInfo Function::type_info;

atomic<size_t> Function::m_next_instance_id(0);

Function::Function(const ResultVector& results,
                   const ParameterVector& parameters,
                   const std::string& name)
    : Lambda(results, parameters)
    , m_name(name)
    , m_unique_name("Function_" + to_string(m_next_instance_id.fetch_add(1)))
    , m_topological_sorter(topological_sort<std::vector<std::shared_ptr<Node>>>)
{
    validate_nodes_and_infer_types();
}

Function::Function(const OutputVector& results,
                   const ParameterVector& parameters,
                   const std::string& name)
    : Lambda(results, parameters)
    , m_name(name)
    , m_unique_name("Function_" + to_string(m_next_instance_id.fetch_add(1)))
    , m_topological_sorter(topological_sort<std::vector<std::shared_ptr<Node>>>)
{
    validate_nodes_and_infer_types();
}

Function::Function(const NodeVector& results,
                   const ParameterVector& parameters,
                   const std::string& name)
    : Lambda(as_output_vector(results), parameters)
    , m_name(name)
    , m_unique_name("Function_" + to_string(m_next_instance_id.fetch_add(1)))
    , m_topological_sorter(topological_sort<std::vector<std::shared_ptr<Node>>>)
{
    validate_nodes_and_infer_types();
}

Function::Function(const std::shared_ptr<Node>& result,
                   const ParameterVector& parameters,
                   const std::string& name)
    : Function(result->outputs(), parameters, name)
{
}

void Function::validate_nodes_and_infer_types()
{
    for (auto& node : get_ordered_ops())
    {
        node->revalidate_and_infer_types();

        // If we find a parameter make sure it is in the list of parameters of the function
        if (op::is_parameter(node))
        {
            auto it = std::find(m_parameters.begin(), m_parameters.end(), node);
            if (it == m_parameters.end())
            {
                throw ngraph_error("Function references undeclared parameter");
            }
        }
    }
}

std::vector<shared_ptr<Node>> Function::get_ordered_ops() const
{
    vector<shared_ptr<Node>> nodes;
    for (auto& r : get_results())
    {
        nodes.push_back(r);
    }
    for (auto& param : get_parameters())
    {
        nodes.push_back(param);
    }

    return m_topological_sorter(nodes);
}

void Function::map_unordered_ops(std::function<void(Node*)> f) const
{
    std::unordered_set<Node*> unordered_ops;
    std::stack<Node*, std::vector<Node*>> remaining_ops;
    for (auto& r : get_results())
    {
        remaining_ops.push(r.get());
    }
    for (auto& param : get_parameters())
    {
        remaining_ops.push(param.get());
    }
    while (remaining_ops.size() > 0)
    {
        Node* op = remaining_ops.top();
        remaining_ops.pop();
        if (unordered_ops.insert(op).second)
        {
            f(op);
            for (size_t i = 0; i < op->get_input_size(); ++i)
            {
                remaining_ops.push(op->get_input_node_ptr(i));
            }
            for (auto& cdep : op->get_control_dependencies())
            {
                remaining_ops.push(cdep.get());
            }
        }
    }
}

const std::string& Function::get_friendly_name() const
{
    if (m_name.empty())
    {
        return m_unique_name;
    }
    return m_name;
}

const std::string& Function::get_name() const
{
    return m_unique_name;
}

void Function::set_friendly_name(const string& name)
{
    if (m_name.empty())
    {
        m_name = name;
    }
    else
    {
        throw ngraph_error("Function name may be set exactly once");
    }
}

std::ostream& operator<<(std::ostream& out, const Function& f)
{
    out << "Function(" << f.get_name() << ")";
    return out;
}

size_t Function::get_output_size() const
{
    return m_results.size();
}

const element::Type& Function::get_output_element_type(size_t i) const
{
    return m_results.at(i)->get_element_type();
}

const Shape& Function::get_output_shape(size_t i) const
{
    return m_results.at(i)->get_shape();
}

const PartialShape& Function::get_output_partial_shape(size_t i) const
{
    return m_results.at(i)->get_output_partial_shape(0);
}

shared_ptr<Node> Function::get_output_op(size_t i) const
{
    return m_results.at(i);
}

Output<Node> Function::output(size_t i) const
{
    return m_results.at(i);
}

shared_ptr<Node> Function::get_result() const
{
    if (m_results.size() != 1)
    {
        throw ngraph_error("get_result() must be called on a function with exactly one result.");
    }
    return m_results.at(0);
}

std::vector<shared_ptr<Node>> Function::get_ops() const
{
    std::vector<std::shared_ptr<Node>> ops;
    traverse_nodes(this, [&](shared_ptr<Node> node) { ops.push_back(node); });
    return ops;
}

void Function::replace_node(std::shared_ptr<Node> old, std::shared_ptr<Node> repl)
{
    ngraph::replace_node(old, repl);
}

size_t Function::get_graph_size() const
{
    size_t total_size = 0;
    for (auto node : get_ops())
    {
        total_size += sizeof(*node);
        if (node->description() == "Constant")
        {
            const Shape& shape = node->get_output_shape(0);
            size_t const_size = node->get_output_element_type(0).size();
            if (shape.size() == 0)
            {
                total_size += const_size;
            }
            else
            {
                total_size += (const_size * shape_size(node->get_output_shape(0)));
            }
        }
    }
    return total_size;
}

// TODO(pthoreho) this will be expensive, since we will be traversing all the nodes in
// the graph, figure out if their is a way to cache the result and invalidate/update
// the result if the function is modified
bool Function::is_dynamic() const
{
    auto list_of_nodes = this->get_ops();
    for (auto& node : list_of_nodes)
    {
        if (node->get_output_partial_shape(0).is_dynamic())
        {
            return true;
        }
    }
    return false;
}

void Function::replace_parameter(size_t parameter_index, const shared_ptr<op::Parameter>& parameter)
{
    NGRAPH_CHECK(parameter_index < m_parameters.size(),
                 "replace_parameter(): Tried to replace parameter at index ",
                 parameter_index,
                 " but the function only has ",
                 m_parameters.size(),
                 " parameters.");
    replace_node(m_parameters[parameter_index], parameter);
    m_parameters[parameter_index] = parameter;
}

void Function::set_topological_sort(topological_sort_t sorter)
{
    m_topological_sorter = sorter;
}
