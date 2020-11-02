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

#include "itt.hpp"
#include "ngraph/factory_adapter.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr DiscreteTypeInfo Function::type_info;

atomic<size_t> Function::m_next_instance_id(0);

Function::Function(const ResultVector& results,
                   const ParameterVector& parameters,
                   const std::string& name)
    : m_results(results)
    , m_parameters(parameters)
    , m_name(name)
    , m_unique_name("Function_" + to_string(m_next_instance_id.fetch_add(1)))
    , m_topological_sorter(topological_sort<std::vector<std::shared_ptr<Node>>>)
{
    validate_nodes_and_infer_types();
}

Function::Function(const OutputVector& results,
                   const ParameterVector& parameters,
                   const std::string& name)
    : m_results(as_result_vector(results))
    , m_parameters(parameters)
    , m_name(name)
    , m_unique_name("Function_" + to_string(m_next_instance_id.fetch_add(1)))
    , m_topological_sorter(topological_sort<std::vector<std::shared_ptr<Node>>>)
{
    validate_nodes_and_infer_types();
}

Function::Function(const NodeVector& results,
                   const ParameterVector& parameters,
                   const std::string& name)
    : m_results(as_result_vector(as_output_vector(results)))
    , m_parameters(parameters)
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

Function::Function(const ResultVector& results,
                   const SinkVector& sinks,
                   const ParameterVector& parameters,
                   const std::string& name)
    : m_results(results)
    , m_sinks(sinks)
    , m_parameters(parameters)
    , m_name(name)
    , m_unique_name("Function_" + to_string(m_next_instance_id.fetch_add(1)))
    , m_topological_sorter(topological_sort<std::vector<std::shared_ptr<Node>>>)
{
    validate_nodes_and_infer_types();
}

Function::Function(const OutputVector& results,
                   const SinkVector& sinks,
                   const ParameterVector& parameters,
                   const std::string& name)
    : Function(as_result_vector(results), sinks, parameters, name)
{
}

void Function::validate_nodes_and_infer_types()
{
    OV_ITT_SCOPED_TASK(itt::domains::nGraph, "Function::validate_nodes_and_infer_types");

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
    OV_ITT_SCOPED_TASK(itt::domains::nGraph, "Function::get_ordered_ops");

    vector<shared_ptr<Node>> nodes;
    for (auto& r : get_results())
    {
        nodes.push_back(r);
    }
    for (auto& r : get_sinks())
    {
        nodes.emplace_back(r);
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
    for (auto& r : get_sinks())
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
    m_name = name;
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

int64_t Function::get_parameter_index(const std::shared_ptr<op::Parameter>& parameter) const
{
    int64_t pos = 0;
    for (auto p : get_parameters())
    {
        if (p == parameter)
        {
            return pos;
        }
        pos++;
    }
    return -1;
}

int64_t Function::get_result_index(const Output<Node>& value) const
{
    int64_t pos = 0;
    if (is_type<op::Result>(value.get_node_shared_ptr()))
    {
        auto result = value.get_node_shared_ptr();
        for (auto r : get_results())
        {
            if (r == result)
            {
                return pos;
            }
            pos++;
        }
    }
    else
    {
        for (auto r : get_results())
        {
            if (r->input_value(0) == value)
            {
                return pos;
            }
            pos++;
        }
    }
    return -1;
}

bool Function::evaluate(const HostTensorVector& output_tensors,
                        const HostTensorVector& input_tensors) const
{
    std::map<RawNodeOutput, HostTensorPtr> value_map;
    for (size_t i = 0; i < m_parameters.size(); ++i)
    {
        value_map[m_parameters.at(i)->output(0)] = input_tensors.at(i);
    }
    OutputVector outputs;
    std::map<RawNodeOutput, HostTensorPtr> output_tensor_map;
    for (size_t i = 0; i < m_results.size(); ++i)
    {
        auto result = m_results.at(i)->output(0);
        output_tensor_map[result] = output_tensors.at(i);
        outputs.push_back(result);
    }
    evaluate_nodes(value_map, output_tensor_map, outputs);
    return true;
}

bool Function::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("parameters", m_parameters);
    visitor.on_attribute("results", m_results);
    return true;
}

void Function::add_sinks(const SinkVector& sinks)
{
    m_sinks.insert(m_sinks.end(), sinks.begin(), sinks.end());
}

void Function::remove_sink(const std::shared_ptr<op::Sink>& sink)
{
    m_sinks.erase(std::remove_if(m_sinks.begin(),
                                 m_sinks.end(),
                                 [&sink](std::shared_ptr<op::Sink>& s) { return s == sink; }),
                  m_sinks.end());
}

void Function::add_results(const ResultVector& results)
{
    m_results.insert(m_results.end(), results.begin(), results.end());
}

void Function::remove_result(const std::shared_ptr<op::Result>& result)
{
    m_results.erase(
        std::remove_if(m_results.begin(),
                       m_results.end(),
                       [&result](std::shared_ptr<op::v0::Result>& r) { return r == result; }),
        m_results.end());
}

constexpr DiscreteTypeInfo AttributeAdapter<shared_ptr<Function>>::type_info;

AttributeAdapter<shared_ptr<Function>>::AttributeAdapter(shared_ptr<Function>& ref)
    : m_ref(ref)
{
}

class NodeAttributeAdapter : public FactoryAttributeAdapter<Node>
{
public:
    using FactoryAttributeAdapter::FactoryAttributeAdapter;
    bool on_start(AttributeVisitor& visitor) override
    {
        // Indicate that there is a node following
        m_id = visitor.get_registered_node_id(m_ref);
        m_set_id = (m_ref == nullptr);
        visitor.on_attribute("id", m_id);
        return m_ref == nullptr || m_id != AttributeVisitor::invalid_node_id;
    }
    bool on_finish(AttributeVisitor&) override
    {
        if (m_set_id && m_ref)
        {
            m_ref->set_friendly_name(m_id);
        }
        return true;
    }
    void visit(AttributeVisitor& visitor, const std::string& id)
    {
        visitor.start_structure(id);
        visitor.on_adapter(id, *this);
        visitor.finish_structure();
    }
    static constexpr DiscreteTypeInfo type_info{"Lambda.NodeAttributeAdapter", 0};
    const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    string m_id;
    bool m_set_id;
};

constexpr DiscreteTypeInfo NodeAttributeAdapter::type_info;

bool AttributeAdapter<shared_ptr<Function>>::visit_attributes(AttributeVisitor& visitor)
{
    if (m_ref->get_results().size() > 0)
    {
        NodeVector serialized_nodes;
        {
            // Start with all nodes not already serialized
            visitor.start_structure("nodes");
            NodeVector results;
            for (auto result : m_ref->get_results())
            {
                results.push_back(result);
            }
            for (auto sink : m_ref->get_sinks())
            {
                results.push_back(sink);
            }

            int64_t i = 0;
            ostringstream index;
            traverse_nodes(
                results, [&i, &index, &visitor, &serialized_nodes](shared_ptr<Node> node) -> void {
                    if (AttributeVisitor::invalid_node_id == visitor.get_registered_node_id(node))
                    {
                        // This node hasn't been seen before
                        visitor.register_node(node);
                        index.str("");
                        index << i++;
                        string id = index.str();
                        NodeAttributeAdapter adapter(node);
                        adapter.visit(visitor, id);
                        serialized_nodes.push_back(node);
                    }
                });
            {
                // Sentinel at end
                index.str("");
                index << i++;
                string id = index.str();
                shared_ptr<Node> null_node;
                NodeAttributeAdapter adapter(null_node);
                adapter.visit(visitor, id);
            }
            visitor.finish_structure();
        }
        {
            // Now do all the edges
            visitor.start_structure("edges");
            int64_t i = 0;
            ostringstream index;
            for (auto node : serialized_nodes)
            {
                for (auto input : node->inputs())
                {
                    index.str("");
                    index << i++;
                    string id = index.str();
                    visitor.start_structure(id);
                    string input_node_id = visitor.get_registered_node_id(node);
                    uint64_t input_index = input.get_index();
                    visitor.on_attribute("input_node", input_node_id);
                    visitor.on_attribute("input_index", input_index);
                    auto output = input.get_source_output();
                    string output_node_id =
                        visitor.get_registered_node_id(output.get_node_shared_ptr());
                    uint64_t output_index = output.get_index();
                    visitor.on_attribute("output_node", output_node_id);
                    visitor.on_attribute("output_index", output_index);
                    visitor.finish_structure();
                }
            }
            {
                // Add a sentinel
                index.str("");
                index << i++;
                string id = index.str();
                visitor.start_structure(id);
                string input_node_id = AttributeVisitor::invalid_node_id;
                visitor.on_attribute("input_node", input_node_id);
                visitor.finish_structure();
            }
            visitor.finish_structure();
        }
        {
            // Control dependencies
            visitor.start_structure("control");
            int64_t i = 0;
            ostringstream index;
            for (auto node : serialized_nodes)
            {
                for (auto control : node->get_control_dependencies())
                {
                    index.str("");
                    index << i++;
                    string id = index.str();
                    visitor.start_structure(id);
                    string node_id = visitor.get_registered_node_id(node);
                    string dependency_id = visitor.get_registered_node_id(control);
                    visitor.on_attribute("node", node_id);
                    visitor.on_attribute("dependency", dependency_id);
                    visitor.finish_structure();
                }
            }
            {
                // Add a sentinel
                index.str("");
                index << i++;
                string id = index.str();
                visitor.start_structure(id);
                string node_id = AttributeVisitor::invalid_node_id;
                visitor.on_attribute("node", node_id);
                visitor.finish_structure();
            }
            visitor.finish_structure();
        }
    }
    else
    {
        NodeVector deserialized_nodes;
        {
            // Read the graph
            visitor.start_structure("nodes");
            int64_t i = 0;
            ostringstream index;
            while (true)
            {
                index.str("");
                index << i++;
                string id = index.str();
                shared_ptr<Node> node;
                NodeAttributeAdapter adapter(node);
                adapter.visit(visitor, id);
                if (node)
                {
                    visitor.register_node(node);
                    deserialized_nodes.push_back(node);
                }
                else
                {
                    break;
                }
            }
            visitor.finish_structure();
        }
        {
            visitor.start_structure("edges");
            // Connect the nodes
            int64_t i = 0;
            ostringstream index;
            bool more_edges = true;
            while (more_edges)
            {
                index.str("");
                index << i++;
                string id = index.str();
                visitor.start_structure(id);
                string input_node_id;
                visitor.on_attribute("input_node", input_node_id);
                if (!input_node_id.empty())
                {
                    shared_ptr<Node> input_node = visitor.get_registered_node(input_node_id);
                    NGRAPH_CHECK(input_node, "input node of edge not known");
                    uint64_t input_index;
                    string output_node_id;
                    uint64_t output_index;
                    visitor.on_attribute("input_index", input_index);
                    visitor.on_attribute("output_node", output_node_id);
                    visitor.on_attribute("output_index", output_index);
                    shared_ptr<Node> output_node = visitor.get_registered_node(output_node_id);
                    NGRAPH_CHECK(output_node, "output_node of edge not known");
                    input_node->set_argument(input_index, output_node->output(output_index));
                }
                else
                {
                    more_edges = false;
                }
                visitor.finish_structure();
            }
            visitor.finish_structure();
        }
        {
            // Control dependencies
            visitor.start_structure("control");
            int64_t i = 0;
            ostringstream index;
            bool more_control = true;
            while (more_control)
            {
                index.str("");
                index << i++;
                string id = index.str();
                visitor.start_structure(id);
                string node_id;
                visitor.on_attribute("node", node_id);
                if (!node_id.empty())
                {
                    shared_ptr<Node> node = visitor.get_registered_node(node_id);
                    NGRAPH_CHECK(node, "node of control edge not known");
                    string dependency_id;
                    visitor.on_attribute("dependency", dependency_id);
                    shared_ptr<Node> dependency = visitor.get_registered_node(dependency_id);
                    NGRAPH_CHECK(dependency, "dependency of control edge not known");
                    node->add_control_dependency(dependency);
                }
                else
                {
                    more_control = false;
                }
                visitor.finish_structure();
            }
            visitor.finish_structure();
        }
        for (auto node : topological_sort(deserialized_nodes))
        {
            node->validate_and_infer_types();
        }
    }

    {
        // Finally visit the object attributes
        visitor.start_structure("value");
        m_ref->visit_attributes(visitor);
        visitor.finish_structure();
    }
    return true;
}
