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

#include "ngraph/lambda.hpp"
#include "ngraph/factory_adapter.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr DiscreteTypeInfo Lambda::type_info;

Lambda::Lambda(const OutputVector& results, const ParameterVector& parameters)
    : Lambda(as_result_vector(results), parameters)
{
}

Lambda::Lambda(const ResultVector& results, const ParameterVector& parameters)
    : m_results(results)
    , m_parameters(parameters)
{
}

int64_t Lambda::get_parameter_index(const std::shared_ptr<op::Parameter>& parameter) const
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

int64_t Lambda::get_result_index(const Output<Node>& value) const
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

bool Lambda::evaluate(const HostTensorVector& output_tensors,
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

bool Lambda::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("parameters", m_parameters);
    visitor.on_attribute("results", m_results);
    return true;
}

constexpr DiscreteTypeInfo AttributeAdapter<shared_ptr<Lambda>>::type_info;

AttributeAdapter<shared_ptr<Lambda>>::AttributeAdapter(shared_ptr<Lambda>& ref)
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

bool AttributeAdapter<shared_ptr<Lambda>>::visit_attributes(AttributeVisitor& visitor)
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
