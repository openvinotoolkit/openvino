// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>

#include "evaluator.hpp"
#include "itt.hpp"
#include "layout_utils.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/meta_data.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/op/util/variable_context.hpp"
#include "openvino/op/util/variable_extension.hpp"
#include "openvino/pass/manager.hpp"
#include "shared_node_info.hpp"
#include "transformations/smart_reshape/smart_reshape.hpp"

using namespace std;

atomic<size_t> ov::Model::m_next_instance_id(0);

namespace {

void check_all_variables_registered(const std::vector<shared_ptr<ov::Node>>& ordered_ops,
                                    const ov::op::util::VariableVector& variables) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::ov_pass, "Model::check_all_variables_registered");
    std::stringstream unregistered_variables;
    for (auto& node : ordered_ops) {
        const auto& variable_op = dynamic_pointer_cast<ov::op::util::VariableExtension>(node);
        if (variable_op &&
            std::find(variables.begin(), variables.end(), variable_op->get_variable()) == variables.end())
            unregistered_variables << variable_op->get_variable_id() << std::endl;
    }
    OPENVINO_ASSERT(unregistered_variables.str().empty(),
                    "Model references undeclared variables: ",
                    unregistered_variables.str());
}

void check_all_parameters_registered(const std::vector<shared_ptr<ov::Node>>& ordered_ops,
                                     const ov::ParameterVector& parameters) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::core, "Model::check_all_parameters_registered");

    std::stringstream unregistered_parameters;
    for (auto& node : ordered_ops) {
        if (ov::op::util::is_parameter(node) &&
            std::find(parameters.begin(), parameters.end(), node) == parameters.end())
            unregistered_parameters << node << std::endl;
    }
    OPENVINO_ASSERT(unregistered_parameters.str().empty(),
                    "Model references undeclared parameters: ",
                    unregistered_parameters.str());
}

ov::op::util::VariableVector auto_detect_variables(const std::vector<std::shared_ptr<ov::Node>>& ordered_ops) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::core, "Model::auto_detect_variables");
    unordered_set<ov::op::util::Variable::Ptr> variables;
    for (const auto& op : ordered_ops) {
        if (const auto& variable_op = dynamic_pointer_cast<ov::op::util::VariableExtension>(op)) {
            variables.insert(variable_op->get_variable());
        }
    }
    return ov::op::util::VariableVector(variables.begin(), variables.end());
}

ov::ParameterVector auto_detect_parameters(const std::vector<std::shared_ptr<ov::Node>>& ordered_ops) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::core, "Model::auto_detect_parameters");
    ov::ParameterVector parameter_vector;
    for (const auto& op : ordered_ops) {
        if (const auto& param = ov::as_type_ptr<ov::op::v0::Parameter>(op)) {
            parameter_vector.push_back(param);
        }
    }
    return parameter_vector;
}

// Check that a Node argument for ctor isn't nullptr.
const std::shared_ptr<ov::Node>& verify_node(const std::shared_ptr<ov::Node>& node) {
    OPENVINO_ASSERT(node != nullptr, "Model is incorrect! Some Node equals to nullptr.");
    return node;
}

std::map<ov::Output<ov::Node>, ov::PartialShape> port_shapes_to_node_shapes(
    ov::Model* model,
    const std::map<size_t, ov::PartialShape>& partial_shapes) {
    std::map<ov::Output<ov::Node>, ov::PartialShape> node_shapes;
    for (const auto& it : partial_shapes) {
        const auto port = model->input(it.first);
        node_shapes[port] = it.second;
    }
    return node_shapes;
}

std::map<ov::Output<ov::Node>, ov::PartialShape> tensor_names_shapes_to_node_shapes(
    ov::Model* model,
    const std::map<std::string, ov::PartialShape>& partial_shapes) {
    std::map<ov::Output<ov::Node>, ov::PartialShape> const_pshape;
    std::unordered_map<ov::Node*, std::string> port_tensor_map;
    for (const auto& it : partial_shapes) {
        const auto port = model->input(it.first);
        if (port_tensor_map.find(port.get_node()) != port_tensor_map.end()) {
            OPENVINO_ASSERT(it.second == const_pshape.at(port),
                            "Tensor with names {'",
                            it.first,
                            "', '",
                            port_tensor_map[port.get_node()],
                            "'} has "
                            "conflicting shapes ",
                            it.second,
                            " and ",
                            const_pshape.at(port),
                            ", but they define the same tensor");
        }
        port_tensor_map[port.get_node()] = it.first;
        const_pshape[port] = it.second;
    }
    return const_pshape;
}

}  // namespace

ov::Model::Model(const ResultVector& results, const ov::ParameterVector& parameters, const std::string& name)
    : m_name(name),
      m_unique_name("Model" + to_string(m_next_instance_id.fetch_add(1))),
      m_topological_sorter(ov::topological_sort<std::vector<std::shared_ptr<ov::Node>>>),
      m_results(results),
      m_parameters(parameters) {
    prerequirements(true, false);
}

ov::Model::Model(const OutputVector& results, const ov::ParameterVector& parameters, const std::string& name)
    : m_name(name),
      m_unique_name("Model" + to_string(m_next_instance_id.fetch_add(1))),
      m_topological_sorter(ov::topological_sort<std::vector<std::shared_ptr<ov::Node>>>),
      m_results(as_result_vector(results)),
      m_parameters(parameters) {
    prerequirements(true, false);
}

ov::Model::Model(const NodeVector& results, const ov::ParameterVector& parameters, const std::string& name)
    : m_name(name),
      m_unique_name("Model" + to_string(m_next_instance_id.fetch_add(1))),
      m_topological_sorter(ov::topological_sort<std::vector<std::shared_ptr<ov::Node>>>),
      m_results(as_result_vector(as_output_vector(results))),
      m_parameters(parameters) {
    prerequirements(true, false);
}

ov::Model::Model(const std::shared_ptr<Node>& result, const ov::ParameterVector& parameters, const std::string& name)
    : Model(verify_node(result)->outputs(), parameters, name) {}

ov::Model::Model(const ov::ResultVector& results,
                 const ov::SinkVector& sinks,
                 const ov::ParameterVector& parameters,
                 const std::string& name)
    : m_name(name),
      m_unique_name("Model" + to_string(m_next_instance_id.fetch_add(1))),
      m_topological_sorter(ov::topological_sort<std::vector<std::shared_ptr<Node>>>),
      m_results(results),
      m_sinks(sinks),
      m_parameters(parameters) {
    prerequirements(true, false);
}

ov::Model::Model(const OutputVector& results,
                 const ov::SinkVector& sinks,
                 const ov::ParameterVector& parameters,
                 const std::string& name)
    : Model(as_result_vector(results), sinks, parameters, name) {}

ov::Model::Model(const ov::ResultVector& results,
                 const ov::SinkVector& sinks,
                 const ov::ParameterVector& parameters,
                 const ov::op::util::VariableVector& variables,
                 const std::string& name)
    : m_name(name),
      m_unique_name("Model" + to_string(m_next_instance_id.fetch_add(1))),
      m_topological_sorter(ov::topological_sort<std::vector<std::shared_ptr<Node>>>),
      m_results(results),
      m_sinks(sinks),
      m_parameters(parameters),
      m_variables(variables) {
    prerequirements(false, false);
}

ov::Model::Model(const OutputVector& results,
                 const ov::SinkVector& sinks,
                 const ov::ParameterVector& parameters,
                 const ov::op::util::VariableVector& variables,
                 const std::string& name)
    : Model(as_result_vector(results), sinks, parameters, variables, name) {}

ov::Model::Model(const ov::OutputVector& results,
                 const ov::ParameterVector& parameters,
                 const ov::op::util::VariableVector& variables,
                 const std::string& name)
    : Model(as_result_vector(results), {}, parameters, variables, name) {}

ov::Model::Model(const ov::ResultVector& results,
                 const ov::ParameterVector& parameters,
                 const ov::op::util::VariableVector& variables,
                 const std::string& name)
    : Model(results, {}, parameters, variables, name) {}

ov::Model::Model(const ov::OutputVector& results, const ov::SinkVector& sinks, const string& name)
    : m_name(name),
      m_unique_name("Model" + to_string(m_next_instance_id.fetch_add(1))),
      m_topological_sorter(ov::topological_sort<std::vector<std::shared_ptr<Node>>>),
      m_results(as_result_vector(results)),
      m_sinks(sinks) {
    prerequirements(true, true);
}

ov::Model::Model(const OutputVector& results, const string& name) : Model(results, ov::SinkVector{}, name) {}

ov::Model::~Model() = default;

void ov::Model::prerequirements(bool detect_variables, bool detect_parameters) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::core, "Model::prerequirements");

    for (const auto& param : m_parameters) {
        OPENVINO_ASSERT(param != nullptr, "Model is incorrect! Some Parameter operation equals to nullptr.");
    }

    for (const auto& result : m_results) {
        OPENVINO_ASSERT(result != nullptr, "Model is incorrect! Some Result operation equals to nullptr.");
    }

    for (const auto& sink : m_sinks) {
        OPENVINO_ASSERT(sink != nullptr, "Model is incorrect! Some Sink operation equals to nullptr.");
    }

    for (const auto& variable : m_variables) {
        OPENVINO_ASSERT(variable != nullptr, "Model is incorrect! Some Variable equals to nullptr.");
    }

    m_shared_rt_info = std::make_shared<SharedRTInfo>();

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

void ov::Model::validate_nodes_and_infer_types() const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::core, "Model::validate_nodes_and_infer_types");

    std::stringstream unregistered_parameters;
    std::stringstream unregistered_variables;
    std::unordered_set<const ov::descriptor::Tensor*> tensors;

    for (auto& node : get_ordered_ops()) {
        node->revalidate_and_infer_types();
        for (const auto& output : node->outputs()) {
            const auto& tensor = output.get_tensor();
            // Skip results outputs tensors because result_input_tensor == result_output_tensor
            if (tensors.count(&tensor))
                continue;
            tensors.insert(&tensor);
        }
        if (op::util::is_parameter(node) &&
            std::find(m_parameters.begin(), m_parameters.end(), node) == m_parameters.end())
            unregistered_parameters << node << std::endl;

        const auto& variable_op = dynamic_pointer_cast<op::util::VariableExtension>(node);
        if (variable_op &&
            std::find(m_variables.begin(), m_variables.end(), variable_op->get_variable()) == m_variables.end())
            unregistered_variables << variable_op->get_variable_id() << std::endl;
    }

    OPENVINO_ASSERT(unregistered_parameters.str().empty(),
                    "Model references undeclared parameters: ",
                    unregistered_parameters.str());

    OPENVINO_ASSERT(unregistered_variables.str().empty(),
                    "Model references undeclared Variables: ",
                    unregistered_variables.str());

    for (const auto& output : outputs()) {
        OPENVINO_ASSERT(ov::layout::utils::is_compatible(ov::layout::get_layout(output), output.get_partial_shape()),
                        "Result '",
                        output,
                        "' with shape ",
                        output.get_partial_shape(),
                        " is incompatible with layout ",
                        ov::layout::get_layout(output).to_string());
    }
}

std::vector<shared_ptr<ov::Node>> ov::Model::get_ordered_ops() const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::core, "Model::get_ordered_ops");
    lock_guard<mutex> lock(m_model_mutex);

    NodeVector nodes;
    if (m_shared_rt_info->get_use_topological_cache()) {
        for (const auto& node : m_cached_ordered_ops) {
            if (auto locked_node = node.lock()) {
                nodes.emplace_back(locked_node);
            }
        }
        return nodes;
    }

    for (const auto& r : get_results()) {
        nodes.emplace_back(r);
    }
    for (auto& r : get_sinks()) {
        nodes.emplace_back(r);
    }
    for (auto& param : get_parameters()) {
        nodes.push_back(param);
    }

    auto order = m_topological_sorter(nodes);

    // Update nodes cache and update all nodes to have shared rt info
    // which belongs to the current Model.
    m_cached_ordered_ops.clear();
    for_each(order.cbegin(), order.cend(), [this](const shared_ptr<Node>& node) {
        m_cached_ordered_ops.push_back(node);
        m_cached_ops.insert(node.get());
        node->insert_info(m_shared_rt_info);
    });
    m_cached_output_names.clear();
    m_cached_op_names.clear();
    m_shared_rt_info->set_use_topological_cache(true);

    return order;
}

void ov::Model::map_unordered_ops(std::function<void(Node*)> f) const {
    std::unordered_set<Node*> unordered_ops;
    std::stack<Node*, std::vector<Node*>> remaining_ops;
    for (auto& r : get_results()) {
        remaining_ops.push(r.get());
    }
    for (auto& r : get_sinks()) {
        remaining_ops.push(r.get());
    }

    for (auto& param : get_parameters()) {
        remaining_ops.push(param.get());
    }
    while (!remaining_ops.empty()) {
        Node* op = remaining_ops.top();
        remaining_ops.pop();
        if (unordered_ops.insert(op).second) {
            f(op);
            for (size_t i = 0; i < op->get_input_size(); ++i) {
                remaining_ops.push(op->get_input_node_ptr(i));
            }
            for (auto& cdep : op->get_control_dependencies()) {
                remaining_ops.push(cdep.get());
            }
        }
    }
}

const std::string& ov::Model::get_friendly_name() const {
    if (m_name.empty()) {
        return m_unique_name;
    }
    return m_name;
}

const std::string& ov::Model::get_name() const {
    return m_unique_name;
}

void ov::Model::set_friendly_name(const string& name) {
    m_name = name;
}

std::ostream& ov::operator<<(std::ostream& out, const ov::Model& f) {
    out << "Model(" << f.get_name() << ")";
    return out;
}

size_t ov::Model::get_output_size() const {
    return m_results.size();
}

const ov::element::Type& ov::Model::get_output_element_type(size_t i) const {
    return m_results.at(i)->get_element_type();
}

const ov::Shape& ov::Model::get_output_shape(size_t i) const {
    return m_results.at(i)->get_shape();
}

const ov::PartialShape& ov::Model::get_output_partial_shape(size_t i) const {
    return m_results.at(i)->get_output_partial_shape(0);
}

shared_ptr<ov::Node> ov::Model::get_output_op(size_t i) const {
    return m_results.at(i);
}

shared_ptr<ov::Node> ov::Model::get_result() const {
    OPENVINO_ASSERT(m_results.size() == 1, "get_result() must be called on a Model with exactly one result.");
    return m_results.at(0);
}

std::vector<shared_ptr<ov::Node>> ov::Model::get_ops() const {
    std::vector<std::shared_ptr<Node>> ops;
    ov::traverse_nodes(this, [&](shared_ptr<Node> node) {
        ops.push_back(node);
    });
    return ops;
}

void ov::Model::replace_node(std::shared_ptr<Node> old, std::shared_ptr<Node> repl) {
    ov::replace_node(old, repl);
}

size_t ov::Model::get_graph_size() const {
    size_t total_size = 0;
    for (auto node : get_ops()) {
        total_size += sizeof(*node);
        if (node->description() == "Constant") {
            const ov::Shape& shape = node->get_output_shape(0);
            size_t const_size = node->get_output_element_type(0).size();
            if (shape.size() == 0) {
                total_size += const_size;
            } else {
                total_size += (const_size * shape_size(node->get_output_shape(0)));
            }
        }
    }
    return total_size;
}

bool ov::Model::is_dynamic() const {
    auto list_of_nodes = this->get_ops();
    for (auto& node : list_of_nodes) {
        if (node->get_output_partial_shape(0).is_dynamic()) {
            return true;
        }
    }
    return false;
}

void ov::Model::replace_parameter(size_t parameter_index, const shared_ptr<ov::op::v0::Parameter>& parameter) {
    OPENVINO_ASSERT(parameter_index < m_parameters.size(),
                    "replace_parameter(): Tried to replace parameter at index ",
                    parameter_index,
                    " but the Model only has ",
                    m_parameters.size(),
                    " parameters.");
    replace_node(m_parameters[parameter_index], parameter);
    m_parameters[parameter_index] = parameter;
}

void ov::Model::set_topological_sort(topological_sort_t sorter) {
    m_topological_sorter = std::move(sorter);
    // reset topological nodes order cache as new sorter can have different behaviour
    m_shared_rt_info->set_use_topological_cache(false);
}

int64_t ov::Model::get_parameter_index(const std::shared_ptr<ov::op::v0::Parameter>& parameter) const {
    int64_t pos = 0;
    for (auto p : get_parameters()) {
        if (p == parameter) {
            return pos;
        }
        pos++;
    }
    return -1;
}

int64_t ov::Model::get_result_index(const Output<Node>& value) const {
    return get_result_index(Output<const Node>(value.get_node(), value.get_index()));
}

int64_t ov::Model::get_result_index(const Output<const Node>& value) const {
    int64_t pos = 0;
    if (is_type<ov::op::v0::Result>(value.get_node_shared_ptr())) {
        auto result = value.get_node_shared_ptr();
        for (auto r : get_results()) {
            if (r == result) {
                return pos;
            }
            pos++;
        }
    } else {
        for (auto r : get_results()) {
            const auto& input_value = r->input_value(0);
            const auto result_input = Output<const Node>(input_value.get_node(), input_value.get_index());
            if (result_input == value) {
                return pos;
            }
            pos++;
        }
    }
    return -1;
}

bool ov::Model::evaluate(ov::TensorVector& output_tensors, const ov::TensorVector& input_tensors) const {
    ov::EvaluationContext evaluation_context;
    return evaluate(output_tensors, input_tensors, evaluation_context);
}

bool ov::Model::evaluate(ov::TensorVector& output_tensors,
                         const ov::TensorVector& input_tensors,
                         ov::EvaluationContext& evaluation_context) const {
    evaluation_context.emplace("VariableContext", ov::op::util::VariableContext());
    std::map<RawNodeOutput, ov::Tensor> value_map;
    OPENVINO_ASSERT(input_tensors.size() == m_parameters.size(),
                    "Cannot evaluate model! Number of tensors (",
                    input_tensors.size(),
                    ") is not equal to number of parameters (",
                    m_parameters.size(),
                    ").");
    for (size_t i = 0; i < m_parameters.size(); ++i) {
        value_map[m_parameters.at(i)->output(0)] = input_tensors.at(i);
        OPENVINO_ASSERT(m_parameters.at(i)->get_partial_shape().is_dynamic() ||
                            m_parameters.at(i)->get_partial_shape().to_shape() == input_tensors[i].get_shape(),
                        "Cannot evaluate model! Tensor input shape and Parameter op with index ",
                        i,
                        " are mismatches.");
    }
    OutputVector outputs;
    std::map<RawNodeOutput, ov::Tensor> output_tensor_map;
    for (size_t i = 0; i < m_results.size(); ++i) {
        auto result = m_results.at(i)->output(0);
        output_tensor_map[result] = output_tensors.at(i);
        outputs.push_back(result);
    }
    for (const auto& m_sink : m_sinks) {
        outputs.push_back(m_sink);
    }
    // evaluate nodes
    Evaluator<Tensor> evaluator({}, value_map);
    evaluator.set_universal_handler(
        [&output_tensor_map, &evaluation_context](Node* node,
                                                  const ov::TensorVector& input_tensors) -> ov::TensorVector {
            ov::TensorVector output_tensors;
            for (const auto& v : node->outputs()) {
                auto it = output_tensor_map.find(v);
                if (it == output_tensor_map.end()) {
                    output_tensors.emplace_back(v);
                } else {
                    output_tensors.push_back(it->second);
                }
            }
            if (node->evaluate(output_tensors, input_tensors, evaluation_context)) {
                for (size_t i = 0; i < node->outputs().size(); i++) {
                    const auto& v = node->output(i);
                    auto it = output_tensor_map.find(v);
                    if (it != output_tensor_map.end()) {
                        it->second = output_tensors[i];
                    }
                }
                return output_tensors;
            } else {
                OPENVINO_THROW("Evaluation failed on ", node);
            }
        });
    for (const auto& value : outputs) {
        evaluator.evaluate(value);
    }
    for (size_t i = 0; i < m_results.size(); ++i) {
        auto result = m_results.at(i)->output(0);
        output_tensors.at(i) = output_tensor_map[result];
    }
    return true;
}

bool ov::Model::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("parameters", m_parameters);
    visitor.on_attribute("results", m_results);
    return true;
}

void ov::Model::add_sinks(const ov::SinkVector& sinks) {
    m_sinks.insert(m_sinks.end(), sinks.begin(), sinks.end());
    for (const auto& sink : sinks) {
        if (const auto& variable_op = dynamic_pointer_cast<op::util::VariableExtension>(sink)) {
            if (find(m_variables.begin(), m_variables.end(), variable_op->get_variable()) == m_variables.end()) {
                m_variables.push_back(variable_op->get_variable());
            }
        }
    }
    // reset topological nodes order cache as new sinks/results/parameters
    // can be in a separate connectivity component.
    m_shared_rt_info->set_use_topological_cache(false);
}

void ov::Model::remove_sink(const std::shared_ptr<ov::op::Sink>& sink) {
    m_sinks.erase(std::remove_if(m_sinks.begin(),
                                 m_sinks.end(),
                                 [&sink](std::shared_ptr<ov::op::Sink>& s) {
                                     return s == sink;
                                 }),
                  m_sinks.end());
    m_shared_rt_info->set_use_topological_cache(false);
}

void ov::Model::add_results(const ResultVector& results) {
    m_results.insert(m_results.end(), results.begin(), results.end());
    // reset topological nodes order cache as new sinks/results/parameters
    // can be in a separate connectivity component.
    m_shared_rt_info->set_use_topological_cache(false);
}

void ov::Model::remove_result(const std::shared_ptr<ov::op::v0::Result>& result) {
    m_results.erase(std::remove_if(m_results.begin(),
                                   m_results.end(),
                                   [&result](std::shared_ptr<ov::op::v0::Result>& r) {
                                       return r == result;
                                   }),
                    m_results.end());
    m_shared_rt_info->set_use_topological_cache(false);
}

void ov::Model::add_parameters(const ov::ParameterVector& params) {
    for (size_t i = 0; i < params.size(); i++) {
        for (size_t j = 0; j < m_parameters.size(); j++) {
            OPENVINO_ASSERT(params[i] != m_parameters[j],
                            "add_parameters(): Tried to add parameter (index in array ",
                            i,
                            ") but Model already have the same parameter with index ",
                            j);
        }
    }
    m_parameters.insert(m_parameters.end(), params.begin(), params.end());
    // reset topological nodes order cache as new sinks/results/parameters
    // can be in a separate connectivity component.
    m_shared_rt_info->set_use_topological_cache(false);
}

void ov::Model::remove_parameter(const std::shared_ptr<ov::op::v0::Parameter>& param) {
    m_parameters.erase(std::remove_if(m_parameters.begin(),
                                      m_parameters.end(),
                                      [&param](std::shared_ptr<ov::op::v0::Parameter>& r) {
                                          return r == param;
                                      }),
                       m_parameters.end());
    m_shared_rt_info->set_use_topological_cache(false);
}

void ov::Model::add_variables(const op::util::VariableVector& variables) {
    m_variables.insert(m_variables.end(), variables.begin(), variables.end());
}

void ov::Model::remove_variable(const op::util::Variable::Ptr& variable) {
    m_variables.erase(std::remove_if(m_variables.begin(),
                                     m_variables.end(),
                                     [&variable](op::util::Variable::Ptr& v) {
                                         return v == variable;
                                     }),
                      m_variables.end());
}

ov::op::util::Variable::Ptr ov::Model::get_variable_by_id(const string& variable_id) const {
    auto variable =
        std::find_if(m_variables.begin(), m_variables.end(), [&variable_id](const ov::op::util::Variable::Ptr& cur) {
            return cur->get_info().variable_id == variable_id;
        });
    if (variable != m_variables.end())
        return *variable;
    else
        return ov::op::util::Variable::Ptr();
}

/// Output Model
std::vector<ov::Output<const ov::Node>> ov::Model::outputs() const {
    std::vector<ov::Output<const ov::Node>> results;
    for (const auto& res : m_results) {
        std::shared_ptr<const ov::Node> result = res;
        results.emplace_back(result);
    }
    return results;
}
ov::Output<const ov::Node> ov::Model::output() const {
    OPENVINO_ASSERT(m_results.size() == 1, "output() must be called on a Model with exactly one result.");
    std::shared_ptr<const ov::Node> result = m_results.at(0);
    return result;
}
ov::Output<const ov::Node> ov::Model::output(size_t i) const {
    std::shared_ptr<const ov::Node> result = m_results.at(i);
    return result;
}
ov::Output<const ov::Node> ov::Model::output(const std::string& tensor_name) const {
    for (const auto& res : m_results) {
        if (res->get_input_tensor(0).get_names().count(tensor_name)) {
            std::shared_ptr<const ov::Node> result = res;
            return result;
        }
    }
    OPENVINO_THROW("Output for tensor name '", tensor_name, "' is not found.");
}

std::vector<ov::Output<ov::Node>> ov::Model::outputs() {
    std::vector<ov::Output<ov::Node>> results;
    for (const auto& result : m_results) {
        results.emplace_back(result);
    }
    return results;
}
ov::Output<ov::Node> ov::Model::output() {
    OPENVINO_ASSERT(m_results.size() == 1, "output() must be called on a Model with exactly one result.");
    return m_results.at(0);
}
ov::Output<ov::Node> ov::Model::output(size_t i) {
    return m_results.at(i);
}
ov::Output<ov::Node> ov::Model::output(const std::string& tensor_name) {
    for (const auto& res : m_results) {
        if (res->get_input_tensor(0).get_names().count(tensor_name))
            return res;
    }
    OPENVINO_THROW("Output for tensor name '", tensor_name, "' is not found.");
}

/// Input Model
std::vector<ov::Output<const ov::Node>> ov::Model::inputs() const {
    std::vector<ov::Output<const ov::Node>> inputs;
    for (const auto& input : m_parameters) {
        std::shared_ptr<const ov::Node> parameter = input;
        inputs.emplace_back(parameter);
    }
    return inputs;
}

ov::Output<const ov::Node> ov::Model::input() const {
    OPENVINO_ASSERT(m_parameters.size() == 1, "input() must be called on a Model with exactly one parameter.");
    std::shared_ptr<const ov::Node> parameter = m_parameters.at(0);
    return parameter;
}
ov::Output<const ov::Node> ov::Model::input(size_t i) const {
    std::shared_ptr<const ov::Node> parameter = m_parameters.at(i);
    return parameter;
}
ov::Output<const ov::Node> ov::Model::input(const std::string& tensor_name) const {
    for (const auto& param : m_parameters) {
        if (param->get_output_tensor(0).get_names().count(tensor_name)) {
            std::shared_ptr<const ov::Node> parameter = param;
            return parameter;
        }
    }
    OPENVINO_THROW("Input for tensor name '", tensor_name, "' is not found.");
}

std::vector<ov::Output<ov::Node>> ov::Model::inputs() {
    std::vector<ov::Output<ov::Node>> inputs;
    for (const auto& input : m_parameters) {
        inputs.emplace_back(input);
    }
    return inputs;
}

ov::Output<ov::Node> ov::Model::input() {
    OPENVINO_ASSERT(m_parameters.size() == 1, "input() must be called on a Model with exactly one parameter.");
    return m_parameters.at(0);
}
ov::Output<ov::Node> ov::Model::input(size_t i) {
    return m_parameters.at(i);
}
ov::Output<ov::Node> ov::Model::input(const std::string& tensor_name) {
    for (const auto& param : m_parameters) {
        if (param->get_output_tensor(0).get_names().count(tensor_name))
            return param;
    }
    OPENVINO_THROW("Input for tensor name '", tensor_name, "' is not found.");
}

void ov::Model::reshape(const ov::PartialShape& partial_shape,
                        const std::unordered_map<std::string, ov::PartialShape>& variable_shapes) {
    OPENVINO_ASSERT(m_parameters.size() == 1, "must be called on a Model with exactly one parameter.");
    std::map<size_t, ov::PartialShape> shapes{{0, partial_shape}};
    reshape(shapes, variable_shapes);
}

void ov::Model::reshape(const std::map<size_t, ov::PartialShape>& partial_shapes,
                        const std::unordered_map<std::string, ov::PartialShape>& variable_shapes) {
    reshape(port_shapes_to_node_shapes(this, partial_shapes), variable_shapes);
}

void ov::Model::reshape(const std::map<std::string, ov::PartialShape>& partial_shapes,
                        const std::unordered_map<std::string, ov::PartialShape>& variable_shapes) {
    reshape(tensor_names_shapes_to_node_shapes(this, partial_shapes), variable_shapes);
}

void ov::Model::reshape(const std::map<ov::Output<ov::Node>, ov::PartialShape>& partial_shapes,
                        const std::unordered_map<std::string, ov::PartialShape>& variables_shapes) {
    if (partial_shapes.empty())
        return;

    const auto& params = get_parameters();
    std::unordered_map<ov::op::v0::Parameter*, ov::PartialShape> new_param_shapes;

    // Check that we need to do reshape only if input shapes will be changed
    bool need_reshape = false;
    for (const auto& partial_shape : partial_shapes) {
        bool shape_is_used = false;

        for (const auto& param : params) {
            const auto port = param->output(0);
            if (port == partial_shape.first) {
                shape_is_used = true;

                if (param->get_output_partial_shape(0).is_dynamic() ||
                    param->get_output_partial_shape(0) != partial_shape.second) {
                    need_reshape = true;
                    new_param_shapes[param.get()] = partial_shape.second;
                }
                break;
            }
        }

        OPENVINO_ASSERT(shape_is_used,
                        "PartialShape for port '",
                        *partial_shape.first.get_node(),
                        "' is not used in ov::Model::reshape");
    }

    if (!need_reshape)
        return;

    std::unordered_map<op::util::Variable*, PartialShape> new_vars_shapes;
    std::unordered_map<op::util::Variable*, PartialShape> original_vars_shapes;
    for (const auto& variable : get_variables()) {
        const auto& var_info = variable->get_info();

        for (const auto& var_id_new_shape : variables_shapes) {
            const auto& variable_id = var_id_new_shape.first;
            const auto& new_shape = var_id_new_shape.second;
            if (variable_id == var_info.variable_id && new_shape != var_info.data_shape) {
                original_vars_shapes[variable.get()] = var_info.data_shape;
                new_vars_shapes[variable.get()] = new_shape;
            }
        }
    }

    // save original parameters shape
    std::unordered_map<ov::op::v0::Parameter*, ov::PartialShape> original_input_shapes;
    for (const auto& param : params) {
        original_input_shapes[param.get()] = param->get_output_partial_shape(0);
    }

    std::unordered_map<op::util::Variable*, PartialShape> original_var_shapes;
    for (const auto& v : get_variables()) {
        original_var_shapes[v.get()] = v->get_info().data_shape;
    }

    auto reshape_only = [this](const std::unordered_map<op::v0::Parameter*, PartialShape>& pshapes,
                               const std::unordered_map<op::util::Variable*, PartialShape>& vars_shapes) {
        for (const auto& pshape : pshapes) {
            pshape.first->set_partial_shape(pshape.second);
        }

        for (const auto& shape : vars_shapes) {
            shape.first->update_data_shape(shape.second);
        }

        validate_nodes_and_infer_types();
    };

    try {
        ov::pass::Manager ssr_manager("SmartReshape");
        ssr_manager.register_pass<ov::pass::SmartReshape>();
        ssr_manager.run_passes(shared_from_this());

        reshape_only(new_param_shapes, new_vars_shapes);
    } catch (...) {
        // restore shapes to original ones
        reshape_only(original_input_shapes, original_vars_shapes);
        throw;
    }
}

ov::Output<ov::Node> ov::Model::add_output(const std::string& tensor_name) {
    auto cache_valid = [&]() {
        return m_cached_output_names.count(tensor_name) &&
               m_cached_output_names[tensor_name].get_names().count(tensor_name) > 0;
    };
    if (!m_shared_rt_info->get_use_topological_cache() || !cache_valid()) {
        m_cached_output_names.clear();
        // get_ordered_ops will update topological cache if necessary
        for (const auto& op : get_ordered_ops()) {
            for (const auto& output : op->outputs()) {
                for (const auto& name : output.get_names()) {
                    m_cached_output_names[name] = output;
                }
            }
        }
    }
    OPENVINO_ASSERT(m_cached_output_names.count(tensor_name),
                    "Model::add_output. Tensor name '",
                    tensor_name + "' was not found.");
    return add_output(m_cached_output_names.at(tensor_name));
}

ov::Output<ov::Node> ov::Model::add_output(const std::string& op_name, size_t output_idx) {
    auto cache_valid = [&]() {
        if (m_cached_op_names.count(op_name)) {
            auto op = m_cached_op_names[op_name].lock();
            return op && op->get_friendly_name() == op_name && op->get_output_size() > output_idx;
        }
        return false;
    };
    if (!m_shared_rt_info->get_use_topological_cache() || !cache_valid()) {
        m_cached_op_names.clear();
        // get_ordered_ops will update topological cache to 'true' if necessary
        for (const auto& op : get_ordered_ops()) {
            m_cached_op_names[op->get_friendly_name()] = op;
        }
    }
    OPENVINO_ASSERT(m_cached_op_names.count(op_name),
                    "Model::add_output. Operation with name '",
                    op_name,
                    "' was not found.");
    auto op = m_cached_op_names[op_name].lock();
    OPENVINO_ASSERT(op, "Model::add_output. Operation with name '", op_name, "' is expired.");
    OPENVINO_ASSERT(output_idx < op->get_output_size(),
                    "Cannot add output to port ",
                    std::to_string(output_idx),
                    " operation ",
                    op->get_friendly_name(),
                    " has only ",
                    std::to_string(op->get_output_size()),
                    " outputs.");

    return add_output(op->output(output_idx));
}

ov::Output<ov::Node> ov::Model::add_output(const ov::Output<ov::Node>& port) {
    auto cache_valid = [&]() {
        return m_cached_ops.count(port.get_node());
    };
    if (ov::op::util::is_output(port.get_node()))
        return port;
    for (const auto& input : port.get_target_inputs()) {
        // Do not add result if port is already connected with result
        if (ov::op::util::is_output(input.get_node())) {
            return input.get_node()->output(0);
        }
    }
    m_results.emplace_back(std::make_shared<ov::op::v0::Result>(port, true));
    auto& result = m_results.back();
    if (m_shared_rt_info->get_use_topological_cache()) {
        if (cache_valid()) {
            // Full update of topological cache is not needed, 'result' can be just inserted to the end
            m_cached_ordered_ops.push_back(result);
            m_cached_ops.insert(result.get());
            result->insert_info(m_shared_rt_info);  // Just for consistency, not required for Result nodes
        } else {
            m_shared_rt_info->set_use_topological_cache(false);
        }
    }
    return result->output(0);
}

std::shared_ptr<ov::Model> ov::Model::clone() const {
    std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>> node_map;
    return ov::clone_ov_model(*this, node_map);
}

bool ov::Model::has_rt_info(const std::vector<std::string>& args) const {
    return has_rt_info(m_rt_info, args.cbegin(), args.cend());
}

bool ov::Model::has_rt_info(const ov::AnyMap& info,
                            const std::vector<std::string>::const_iterator& begin,
                            const std::vector<std::string>::const_iterator& end) const {
    if (!has_rt_arg(info, *begin))
        return false;
    if (begin == end - 1) {
        return true;
    } else {
        return has_rt_info(get_map_from_attr(get_rt_arg<std::string>(info, *begin)), begin + 1, end);
    }
}

ov::Any& ov::Model::get_rt_info(ov::AnyMap& info,
                                const std::vector<std::string>::const_iterator& begin,
                                const std::vector<std::string>::const_iterator& end) {
    if (begin == end - 1) {
        return get_rt_arg(info, *begin);
    } else {
        ov::Any& rt_attr = get_rt_arg<std::string>(info, *begin);
        return get_rt_info(get_map_from_attr(rt_attr), begin + 1, end);
    }
}

// Allow to get constant attribute for the vector
const ov::Any& ov::Model::get_rt_info(const ov::AnyMap& info,
                                      const std::vector<std::string>::const_iterator& begin,
                                      const std::vector<std::string>::const_iterator& end) const {
    if (begin == end - 1) {
        return get_rt_arg(info, *begin);
    } else {
        const ov::Any& rt_attr = get_rt_arg<std::string>(info, *begin);
        return get_rt_info(get_map_from_attr(rt_attr), begin + 1, end);
    }
}

const ov::AnyMap& ov::Model::get_map_from_attr(const ov::Any& info) const {
    // lock to get meta from different threads in order to avoid thread safety
    // implementations of meta information for each frontend
    std::lock_guard<mutex> lock(m_model_mutex);
    if (info.is<ov::AnyMap>()) {
        return info.as<ov::AnyMap>();
    } else if (info.is<std::shared_ptr<ov::Meta>>()) {
        std::shared_ptr<ov::Meta> meta = info.as<std::shared_ptr<ov::Meta>>();
        return *info.as<std::shared_ptr<ov::Meta>>();
    }
    OPENVINO_THROW("Cannot get runtime attribute. Path to runtime attribute is incorrect.");
}

ov::AnyMap& ov::Model::get_map_from_attr(ov::Any& info) const {
    // lock to get meta from different threads in order to avoid thread safety
    // implementations of meta information for each frontend
    std::lock_guard<mutex> lock(m_model_mutex);
    if (info.empty()) {
        info = ov::AnyMap();
    }
    if (info.is<ov::AnyMap>()) {
        return info.as<ov::AnyMap>();
    } else if (info.is<std::shared_ptr<ov::Meta>>()) {
        std::shared_ptr<ov::Meta> meta = info.as<std::shared_ptr<ov::Meta>>();
        return *info.as<std::shared_ptr<ov::Meta>>();
    }
    OPENVINO_THROW("Cannot get runtime attribute. Path to runtime attribute is incorrect.");
}

const ov::Any& ov::Model::get_attr(const ov::Any& info) const {
    // lock to get meta from different threads in order to avoid thread safety
    // implementations of meta information for each frontend
    std::lock_guard<mutex> lock(m_model_mutex);
    if (info.is<std::shared_ptr<ov::Meta>>()) {
        std::shared_ptr<ov::Meta> meta = info.as<std::shared_ptr<ov::Meta>>();
        ov::AnyMap& map = *info.as<std::shared_ptr<ov::Meta>>();
        const_cast<ov::Any&>(info) = map;
    }
    return info;
}

ov::Any& ov::Model::get_attr(ov::Any& info) const {
    // lock to get meta from different threads in order to avoid thread safety
    // implementations of meta information for each frontend
    std::lock_guard<mutex> lock(m_model_mutex);
    if (info.empty()) {
        info = ov::AnyMap();
    }
    if (info.is<std::shared_ptr<ov::Meta>>()) {
        std::shared_ptr<ov::Meta> meta = info.as<std::shared_ptr<ov::Meta>>();
        ov::AnyMap& map = *info.as<std::shared_ptr<ov::Meta>>();
        info = map;
    }
    return info;
}

namespace bs_util {
static int64_t get_batch(const ov::Layout& layout, const ov::PartialShape& shape) {
    auto batch_idx = ov::layout::batch_idx(layout);
    if (batch_idx < 0) {
        batch_idx += static_cast<int64_t>(shape.rank().get_length());
    }
    return batch_idx;
}

static void dump_parameter(std::ostream& stream, const std::shared_ptr<const ov::Model>& f, size_t index) {
    const auto& p = f->get_parameters()[index];
    const auto& node = f->input(index);
    stream << index << ": { ";
    if (!node.get_tensor().get_names().empty()) {
        stream << "name='" << node.get_tensor().get_any_name() << "', ";
    }
    stream << "shape=" << node.get_partial_shape();
    if (node.get_partial_shape().rank().is_static()) {
        stream << ", layout=" << p->get_layout().to_string();
        if (!ov::layout::has_batch(p->get_layout())) {
            stream << ", no batch specified";
        } else {
            stream << ", batch="
                   << node.get_partial_shape()[bs_util::get_batch(p->get_layout(), node.get_partial_shape())];
        }
        stream << " }" << std::endl;
    }
}
}  // namespace bs_util

ov::Dimension ov::get_batch(const std::shared_ptr<const ov::Model>& f) {
    bool batch_initialized = false;
    auto batch_size = ov::Dimension::dynamic();
    std::vector<size_t> merged_indexes;
    merged_indexes.reserve(f->inputs().size());
    for (size_t i = 0; i < f->get_parameters().size(); ++i) {
        const auto& param = f->get_parameters()[i];
        const auto& layout = param->get_layout();
        if (!ov::layout::has_batch(layout))
            continue;
        const auto& pshape = param->get_partial_shape();
        if (pshape.rank().is_dynamic()) {
            continue;  // Parameter with fully dynamic rank can't conflict
        }
        auto batch_idx = bs_util::get_batch(layout, pshape);
        if (!Dimension::merge(batch_size, batch_size, pshape[batch_idx])) {
            merged_indexes.push_back(i);
            // Not all dimensions can be merged
            std::stringstream stream;
            stream << "Get original batch size fails due to conflicting batch values for inputs:" << std::endl;
            for (size_t j = 0; j < merged_indexes.size(); ++j) {
                bs_util::dump_parameter(stream, f, merged_indexes[j]);
            }
            stream << "---" << std::endl;
            stream << "Please ensure that N(Batch) dimension is set correctly for listed parameters";
            OPENVINO_ASSERT(false, stream.str());
        } else {
            merged_indexes.push_back(i);
        }
        batch_initialized = true;
    }
    if (!batch_initialized) {
        // Create graceful message to set layout for some parameters
        std::stringstream stream;
        stream << "Get original batch size fails due to batch is not set in any layout for any input. ";
        stream << "Available inputs:" << std::endl;
        for (size_t i = 0; i < f->get_parameters().size(); ++i) {
            bs_util::dump_parameter(stream, f, i);
        }
        stream << "---" << std::endl;
        stream << "Please use 'set_layout' API to set layout with batch dimension, e.g. "
                  "`Model->get_parameters()[index]->set_layout(\"NCHW\");`";

        OPENVINO_ASSERT(false, stream.str());
    }
    return batch_size;
}

void ov::set_batch(const std::shared_ptr<ov::Model>& f, ov::Dimension batch_size) {
    get_batch(f);  // Ensure that function's batch size is valid and can be changed
    std::map<ov::Output<ov::Node>, ov::PartialShape> new_shapes_map;
    // Now batch size can be set for all needed parameters
    for (size_t i = 0; i < f->get_parameters().size(); ++i) {
        const auto& param = f->get_parameters()[i];
        const auto& layout = param->get_layout();
        if (!ov::layout::has_batch(layout))
            continue;
        const auto& pshape = param->get_partial_shape();
        if (pshape.rank().is_dynamic()) {
            continue;  // Parameter with fully dynamic rank can be left as is
        }
        auto batch_idx = bs_util::get_batch(layout, pshape);
        auto new_shape = param->get_partial_shape();
        new_shape[batch_idx] = batch_size;
        new_shapes_map[f->input(i)] = std::move(new_shape);
    }
    try {
        f->reshape(new_shapes_map);
    } catch (const std::exception& e) {
        std::stringstream stream;
        stream << "Failed to set batch size to " << batch_size << ". Possible reasons are:" << std::endl;
        stream << "    1) Ensure that all inputs have valid layout set with batch dimension" << std::endl;
        stream << "    2) Check model's documentation if batch size can be set to it at all" << std::endl;
        stream << "Available inputs:" << std::endl;
        for (size_t i = 0; i < f->get_parameters().size(); ++i) {
            bs_util::dump_parameter(stream, f, i);
            if (new_shapes_map.count(f->input(i))) {
                stream << i << ": Tried reshape " << f->input(i).get_partial_shape() << " to "
                       << new_shapes_map[f->input(i)] << std::endl;
            } else {
                stream << i << ": No reshape has been applied" << std::endl;
            }
        }
        stream << "---" << std::endl;
        stream << "Original error message is: " << e.what();
        OPENVINO_ASSERT(false, stream.str());
    }
}

ov::AttributeAdapter<std::shared_ptr<ov::Model>>::~AttributeAdapter() = default;
