// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <list>
#include <memory>
#include <ngraph/ops.hpp>

#include "itt.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/op/util/variable_context.hpp"
#include "ngraph/op/util/variable_extension.hpp"
#include "ngraph/opsets/opset7.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr DiscreteTypeInfo Function::type_info;

atomic<size_t> Function::m_next_instance_id(0);

void check_all_variables_registered(const std::vector<shared_ptr<Node>>& ordered_ops,
                                    const VariableVector& variables)
{
    OV_ITT_SCOPED_TASK(ngraph::itt::domains::nGraphPass_LT,
                       "Function::check_all_variables_registered");
    std::stringstream unregistered_variables;
    for (auto& node : ordered_ops)
    {
        const auto& variable_op = dynamic_pointer_cast<VariableExtension>(node);
        if (variable_op &&
            std::find(variables.begin(), variables.end(), variable_op->get_variable()) ==
                variables.end())
            unregistered_variables << variable_op->get_variable_id() << std::endl;
    }
    if (!unregistered_variables.str().empty())
        throw ngraph_error("Function references undeclared variables: " +
                           unregistered_variables.str());
}

void check_all_parameters_registered(const std::vector<shared_ptr<Node>>& ordered_ops,
                                     const ParameterVector& parameters)
{
    OV_ITT_SCOPED_TASK(ngraph::itt::domains::nGraph, "Function::check_all_parameters_registered");

    std::stringstream unregistered_parameters;
    for (auto& node : ordered_ops)
    {
        if (op::is_parameter(node) &&
            std::find(parameters.begin(), parameters.end(), node) == parameters.end())
            unregistered_parameters << node << std::endl;
    }
    if (!unregistered_parameters.str().empty())
        throw ngraph_error("Function references undeclared parameters: " +
                           unregistered_parameters.str());
}

VariableVector auto_detect_variables(const std::vector<std::shared_ptr<Node>>& ordered_ops)
{
    OV_ITT_SCOPED_TASK(ngraph::itt::domains::nGraph, "Function::auto_detect_variables");
    unordered_set<VariablePtr> variables;
    for (const auto& op : ordered_ops)
    {
        if (const auto& variable_op = dynamic_pointer_cast<VariableExtension>(op))
        {
            variables.insert(variable_op->get_variable());
        }
    }
    return VariableVector(variables.begin(), variables.end());
}

ParameterVector auto_detect_parameters(const std::vector<std::shared_ptr<Node>>& ordered_ops)
{
    OV_ITT_SCOPED_TASK(ngraph::itt::domains::nGraph, "Function::auto_detect_parameters");
    ParameterVector parameter_vector;
    for (const auto& op : ordered_ops)
    {
        if (const auto& param = dynamic_pointer_cast<opset7::Parameter>(op))
        {
            parameter_vector.push_back(param);
        }
    }
    return parameter_vector;
}

Function::Function(const ResultVector& results,
                   const ParameterVector& parameters,
                   const std::string& name)
    : m_name(name)
    , m_unique_name("Function_" + to_string(m_next_instance_id.fetch_add(1)))
    , m_topological_sorter(topological_sort<std::vector<std::shared_ptr<Node>>>)
    , m_results(results)
    , m_parameters(parameters)
{
    prerequirements(true, false);
}

Function::Function(const OutputVector& results,
                   const ParameterVector& parameters,
                   const std::string& name)
    : m_name(name)
    , m_unique_name("Function_" + to_string(m_next_instance_id.fetch_add(1)))
    , m_topological_sorter(topological_sort<std::vector<std::shared_ptr<Node>>>)
    , m_results(as_result_vector(results))
    , m_parameters(parameters)
{
    prerequirements(true, false);
}

Function::Function(const NodeVector& results,
                   const ParameterVector& parameters,
                   const std::string& name)
    : m_name(name)
    , m_unique_name("Function_" + to_string(m_next_instance_id.fetch_add(1)))
    , m_topological_sorter(topological_sort<std::vector<std::shared_ptr<Node>>>)
    , m_results(as_result_vector(as_output_vector(results)))
    , m_parameters(parameters)
{
    prerequirements(true, false);
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
    : m_name(name)
    , m_unique_name("Function_" + to_string(m_next_instance_id.fetch_add(1)))
    , m_topological_sorter(topological_sort<std::vector<std::shared_ptr<Node>>>)
    , m_results(results)
    , m_sinks(sinks)
    , m_parameters(parameters)
{
    prerequirements(true, false);
}

Function::Function(const OutputVector& results,
                   const SinkVector& sinks,
                   const ParameterVector& parameters,
                   const std::string& name)
    : Function(as_result_vector(results), sinks, parameters, name)
{
}

Function::Function(const ResultVector& results,
                   const SinkVector& sinks,
                   const ParameterVector& parameters,
                   const VariableVector& variables,
                   const std::string& name)
    : m_name(name)
    , m_unique_name("Function_" + to_string(m_next_instance_id.fetch_add(1)))
    , m_topological_sorter(topological_sort<std::vector<std::shared_ptr<Node>>>)
    , m_results(results)
    , m_sinks(sinks)
    , m_parameters(parameters)
    , m_variables(variables)
{
    prerequirements(false, false);
}

Function::Function(const OutputVector& results,
                   const SinkVector& sinks,
                   const ParameterVector& parameters,
                   const VariableVector& variables,
                   const std::string& name)
    : Function(as_result_vector(results), sinks, parameters, variables, name)
{
}

Function::Function(const OutputVector& results,
                   const ParameterVector& parameters,
                   const VariableVector& variables,
                   const std::string& name)
    : Function(as_result_vector(results), {}, parameters, variables, name)
{
}

Function::Function(const ResultVector& results,
                   const ParameterVector& parameters,
                   const VariableVector& variables,
                   const std::string& name)
    : Function(results, {}, parameters, variables, name)
{
}

Function::Function(const OutputVector& results, const SinkVector& sinks, const string& name)
    : m_name(name)
    , m_unique_name("Function_" + to_string(m_next_instance_id.fetch_add(1)))
    , m_topological_sorter(topological_sort<std::vector<std::shared_ptr<Node>>>)
    , m_results(as_result_vector(results))
    , m_sinks(sinks)
{
    prerequirements(true, true);
}

Function::Function(const OutputVector& results, const string& name)
    : Function(results, SinkVector{}, name)
{
}

void Function::prerequirements(bool detect_variables, bool detect_parameters)
{
    OV_ITT_SCOPED_TASK(ngraph::itt::domains::nGraph, "Function::prerequirements");

    const auto& ordered_ops = get_ordered_ops();
    if (detect_parameters)
        m_parameters = auto_detect_parameters(ordered_ops);
    else
        check_all_parameters_registered(ordered_ops, m_parameters);

    if (detect_variables)
        m_variables = auto_detect_variables(ordered_ops);
    else
        check_all_variables_registered(ordered_ops, m_variables);
}

void Function::validate_nodes_and_infer_types() const
{
    OV_ITT_SCOPED_TASK(ngraph::itt::domains::nGraph, "Function::validate_nodes_and_infer_types");

    struct Counter
    {
        int cnt_assign = 0;
        int cnt_read_val = 0;
    };
    std::map<Variable*, Counter> pair_checker;
    std::stringstream unregistered_parameters;
    std::stringstream unregistered_variables;
    for (auto& node : get_ordered_ops())
    {
        node->revalidate_and_infer_types();
        if (op::is_parameter(node) &&
            std::find(m_parameters.begin(), m_parameters.end(), node) == m_parameters.end())
            unregistered_parameters << node << std::endl;

        const auto& variable_op = dynamic_pointer_cast<VariableExtension>(node);
        if (variable_op &&
            std::find(m_variables.begin(), m_variables.end(), variable_op->get_variable()) ==
                m_variables.end())
            unregistered_variables << variable_op->get_variable_id() << std::endl;

        if (const auto& assign = std::dynamic_pointer_cast<op::AssignBase>(node))
        {
            pair_checker[assign->get_variable().get()].cnt_assign++;
        }
        else if (const auto& read_value = std::dynamic_pointer_cast<op::ReadValueBase>(node))
        {
            pair_checker[read_value->get_variable().get()].cnt_read_val++;
        }
    }
    if (!unregistered_parameters.str().empty())
        throw ngraph_error("Function references undeclared parameters: " +
                           unregistered_parameters.str());

    if (!unregistered_variables.str().empty())
        throw ngraph_error("Function references undeclared Variables: " +
                           unregistered_variables.str());
    bool only_pairs = std::all_of(
        pair_checker.begin(), pair_checker.end(), [](const std::pair<Variable*, Counter>& val) {
            return val.second.cnt_assign == 1 && val.second.cnt_read_val == 1;
        });
    if (!only_pairs)
        throw ngraph_error(
            "Function is incorrect. Assign and ReadValue operations must be in pairs on the "
            "network.");
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
    while (!remaining_ops.empty())
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
                        const HostTensorVector& input_tensors,
                        EvaluationContext evaluation_context) const
{
    if (evaluation_context.find("VariableContext") == evaluation_context.end())
        evaluation_context["VariableContext"] =
            std::make_shared<VariantWrapper<VariableContext>>(VariableContext());
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
    for (const auto& m_sink : m_sinks)
    {
        outputs.push_back(m_sink);
    }
    evaluate_nodes(value_map, output_tensor_map, outputs, evaluation_context);
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
    for (const auto& sink : sinks)
    {
        if (const auto& variable_op = dynamic_pointer_cast<VariableExtension>(sink))
        {
            if (find(m_variables.begin(), m_variables.end(), variable_op->get_variable()) ==
                m_variables.end())
            {
                m_variables.push_back(variable_op->get_variable());
            }
        }
    }
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

void Function::add_parameters(const ParameterVector& params)
{
    for (size_t i = 0; i < params.size(); i++)
    {
        for (size_t j = 0; j < m_parameters.size(); j++)
        {
            NGRAPH_CHECK(params[i] != m_parameters[j],
                         "add_parameters(): Tried to add parameter (index in array ",
                         i,
                         ") but function already have the same parameter with index ",
                         j);
        }
    }
    m_parameters.insert(m_parameters.end(), params.begin(), params.end());
}

void Function::remove_parameter(const std::shared_ptr<op::Parameter>& param)
{
    m_parameters.erase(
        std::remove_if(m_parameters.begin(),
                       m_parameters.end(),
                       [&param](std::shared_ptr<op::v0::Parameter>& r) { return r == param; }),
        m_parameters.end());
}

void Function::add_variables(const VariableVector& variables)
{
    m_variables.insert(m_variables.end(), variables.begin(), variables.end());
}

void Function::remove_variable(const VariablePtr& variable)
{
    m_variables.erase(std::remove_if(m_variables.begin(),
                                     m_variables.end(),
                                     [&variable](VariablePtr& v) { return v == variable; }),
                      m_variables.end());
}

VariablePtr Function::get_variable_by_id(const string& variable_id) const
{
    auto variable = std::find_if(
        m_variables.begin(), m_variables.end(), [&variable_id](const VariablePtr& cur) {
            return cur->get_info().variable_id == variable_id;
        });
    if (variable != m_variables.end())
        return *variable;
    else
        return VariablePtr();
}

constexpr DiscreteTypeInfo AttributeAdapter<shared_ptr<Function>>::type_info;
