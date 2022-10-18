// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "int_executable.hpp"

#include <cstring>
#include <openvino/op/util/variable_context.hpp>

#include "evaluates_map.hpp"
#include "ngraph/except.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/type/bfloat16.hpp"
#include "ngraph/type/float16.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_SUPPRESS_DEPRECATED_START

class TemporaryOverrideOutputs {
    std::shared_ptr<Node> node;
    std::vector<PartialShape> orig_shapes;

public:
    TemporaryOverrideOutputs(std::shared_ptr<Node> node, const std::vector<std::shared_ptr<HostTensor>>& args)
        : node(node) {
        for (size_t i = 0; i < args.size(); ++i) {
            auto output = node->get_input_source_output(i);
            orig_shapes.push_back(output.get_partial_shape());
            output.get_tensor().set_partial_shape(args[i]->get_shape());
        }
    }

    ~TemporaryOverrideOutputs() {
        for (size_t i = 0; i < orig_shapes.size(); ++i) {
            auto output = node->get_input_source_output(i);
            output.get_tensor().set_partial_shape(orig_shapes[i]);
        }
    }
};

runtime::interpreter::INTExecutable::INTExecutable(const shared_ptr<Function>& function,
                                                   bool enable_performance_collection)
    : m_is_compiled{true},
      m_performance_counters_enabled{enable_performance_collection} {
    m_function = clone_function(*function);
    for (auto node : m_function->get_ordered_ops()) {
        m_nodes.push_back(node);
    }
    set_parameters_and_results(*m_function);
}

bool runtime::interpreter::INTExecutable::call(const vector<shared_ptr<runtime::Tensor>>& outputs,
                                               const vector<shared_ptr<runtime::Tensor>>& inputs) {
    // convert inputs to HostTensor
    vector<shared_ptr<HostTensor>> func_inputs;
    for (const auto& tensor : inputs) {
        auto host_tensor = static_pointer_cast<runtime::HostTensor>(tensor);
        func_inputs.push_back(host_tensor);
    }
    if (m_nan_check_enabled) {
        perform_nan_check(func_inputs);
    }

    // convert outputs to HostTensor
    vector<shared_ptr<HostTensor>> func_outputs;
    for (const auto& tensor : outputs) {
        auto host_tensor = static_pointer_cast<runtime::HostTensor>(tensor);
        func_outputs.push_back(host_tensor);
    }

    // map function params -> HostTensor
    std::unordered_map<std::shared_ptr<ov::descriptor::Tensor>, shared_ptr<HostTensor>> tensor_map;
    size_t input_count = 0;
    for (const auto& param : get_parameters()) {
        for (size_t i = 0; i < param->get_output_size(); ++i) {
            auto tensor = param->output(i).get_tensor_ptr();
            tensor_map.insert({tensor, func_inputs[input_count++]});
        }
    }

    std::unordered_map<std::shared_ptr<ov::descriptor::Tensor>, size_t> results_map;
    // map function outputs -> HostTensor
    for (size_t output_count = 0; output_count < get_results().size(); ++output_count) {
        auto output = get_results()[output_count]->output(0).get_tensor_ptr();
        results_map.emplace(output, results_map.size());
    }

    EvaluationContext eval_context;
    ov::op::util::VariableContext variable_context;
    eval_context.emplace("VariableContext", variable_context);

    // for each ordered op in the graph
    for (const auto& op : m_nodes) {
        if (dynamic_pointer_cast<op::Parameter>(op) != nullptr) {
            continue;
        }

        // get op inputs from map
        vector<shared_ptr<HostTensor>> op_inputs;
        for (auto input : op->inputs()) {
            auto tensor = input.get_tensor_ptr();
            op_inputs.push_back(tensor_map.at(tensor));
        }

        TemporaryOverrideOutputs overrider(op, op_inputs);
        OutputVector outputs;
        for (size_t i = 0; i < op->inputs().size(); ++i) {
            outputs.push_back(op->get_input_source_output(i));
        }
        auto cloned_node = op->clone_with_new_inputs(outputs);

        // get op outputs from map or create
        vector<shared_ptr<HostTensor>> op_outputs;
        for (size_t i = 0; i < op->get_output_size(); ++i) {
            auto tensor = op->output(i).get_tensor_ptr();
            shared_ptr<HostTensor> host_tensor;
            auto it = tensor_map.find(tensor);
            if (op::is_output(op)) {
                host_tensor = func_outputs[results_map[tensor]];
            } else if (it == tensor_map.end()) {
                // Use cloned_node to create HostTensor with static dimensions
                host_tensor = make_shared<HostTensor>(cloned_node->output(i));
                tensor_map.insert({tensor, host_tensor});
            } else {
                host_tensor = it->second;
            }
            op_outputs.push_back(host_tensor);
        }

        // get op type
        element::Type type;
        if (ov::is_type<op::Convert>(op) || ov::is_type<op::v0::PriorBox>(op) || ov::is_type<op::v8::PriorBox>(op)) {
            type = op->get_input_element_type(0);
        } else if (ov::is_type<op::v1::Equal>(op) || ov::is_type<op::v1::Greater>(op) ||
                   ov::is_type<op::v1::GreaterEqual>(op) || ov::is_type<op::v1::Less>(op) ||
                   ov::is_type<op::v1::LessEqual>(op) || ov::is_type<op::v1::NotEqual>(op)) {
            // Get the type of the second input, not the first
            // All BinaryElementwiseComparision ops have the same type for inputs
            // Select has bool for first input and the type we are interested in for the second
            type = op->get_input_element_type(1);
        } else {
            type = op->get_output_element_type(0);
        }

        if (m_performance_counters_enabled) {
            m_timer_map[op].start();
        }

        if (auto var_extension = std::dynamic_pointer_cast<ov::op::util::VariableExtension>(cloned_node)) {
            auto variable = var_extension->get_variable();
            if (!variable_context.get_variable_value(variable)) {
                auto h_tensor = std::make_shared<ngraph::HostTensor>(cloned_node->get_input_element_type(0),
                                                                     cloned_node->get_input_shape(0));
                h_tensor->write(h_tensor->get_data_ptr(), h_tensor->get_size_in_bytes());
                variable_context.set_variable_value(variable, std::make_shared<VariableValue>(h_tensor));
            }
        }

        // Call evaluate for cloned_node with static shapes
        if (!cloned_node->evaluate(op_outputs, op_inputs, eval_context)) {
            evaluate_node(cloned_node, op_outputs, op_inputs);
        }
        if (m_performance_counters_enabled) {
            m_timer_map[op].stop();
        }
        if (m_nan_check_enabled) {
            perform_nan_check(op_outputs, op.get());
        }
    }

    return true;
}

vector<runtime::PerformanceCounter> runtime::interpreter::INTExecutable::get_performance_data() const {
    vector<runtime::PerformanceCounter> rc;
    for (const pair<shared_ptr<const Node>, stopwatch> p : m_timer_map) {
        rc.emplace_back(p.first, p.second.get_total_microseconds(), p.second.get_call_count());
    }
    return rc;
}

void runtime::interpreter::INTExecutable::perform_nan_check(const vector<shared_ptr<HostTensor>>& tensors,
                                                            const Node* op) {
    size_t arg_number = 1;
    for (const shared_ptr<HostTensor>& tensor : tensors) {
        const element::Type& type = tensor->get_element_type();
        if (type == element::f32) {
            const float* data = tensor->get_data_ptr<float>();
            for (size_t i = 0; i < tensor->get_element_count(); i++) {
                if (std::isnan(data[i])) {
                    if (op) {
                        throw runtime_error("nan found in op '" + op->get_name() + "' output");
                    } else {
                        throw runtime_error("nan found in function's input tensor number " + to_string(arg_number));
                    }
                }
            }
        } else if (type == element::f64) {
            const double* data = tensor->get_data_ptr<double>();
            for (size_t i = 0; i < tensor->get_element_count(); i++) {
                if (std::isnan(data[i])) {
                    if (op) {
                        throw runtime_error("nan found in op '" + op->get_name() + "' output");
                    } else {
                        throw runtime_error("nan found in function's input tensor number " + to_string(arg_number));
                    }
                }
            }
        }
        arg_number++;
    }
}

shared_ptr<ngraph::op::Parameter> runtime::interpreter::INTExecutable::get_parameter(size_t index) const {
    const ParameterVector& parameters = get_parameters();
    NGRAPH_CHECK(index < parameters.size(), "create_tensor for input out of bounds");
    return parameters[index];
}

shared_ptr<ngraph::op::Result> runtime::interpreter::INTExecutable::get_result(size_t index) const {
    const ResultVector& results = get_results();
    NGRAPH_CHECK(index < results.size(), "create_tensor for input out of bounds");
    return results[index];
}
shared_ptr<runtime::Tensor> runtime::interpreter::INTExecutable::create_input_tensor(size_t input_index) {
    shared_ptr<op::Parameter> parameter = get_parameter(input_index);
    return make_shared<runtime::HostTensor>(parameter->get_element_type(), parameter->get_shape());
}

shared_ptr<runtime::Tensor> runtime::interpreter::INTExecutable::create_output_tensor(size_t output_index) {
    shared_ptr<op::Result> result = get_result(output_index);
    return make_shared<runtime::HostTensor>(result->get_element_type(), result->get_shape());
}

vector<shared_ptr<runtime::Tensor>> runtime::interpreter::INTExecutable::create_input_tensor(size_t input_index,
                                                                                             size_t pipeline_depth) {
    vector<shared_ptr<runtime::HostTensor>> tensors;
    shared_ptr<op::Parameter> parameter = get_parameter(input_index);
    for (size_t i = 0; i < pipeline_depth; i++) {
        shared_ptr<runtime::HostTensor> tensor;
        auto t = make_shared<runtime::HostTensor>(parameter->get_element_type(), parameter->get_shape());
        tensor = static_pointer_cast<runtime::HostTensor>(t);
        tensors.push_back(tensor);
    }
    vector<shared_ptr<runtime::Tensor>> result_tensors;
    for (const shared_ptr<runtime::HostTensor>& tensor : tensors) {
        result_tensors.push_back(tensor);
    }
    return result_tensors;
}

vector<shared_ptr<runtime::Tensor>> runtime::interpreter::INTExecutable::create_output_tensor(size_t output_index,
                                                                                              size_t pipeline_depth) {
    vector<shared_ptr<runtime::HostTensor>> tensors;
    shared_ptr<op::Result> result = get_result(output_index);
    for (size_t i = 0; i < pipeline_depth; i++) {
        shared_ptr<runtime::HostTensor> tensor;
        auto t = make_shared<runtime::HostTensor>(result->get_element_type(), result->get_shape());
        tensor = static_pointer_cast<runtime::HostTensor>(t);
        tensors.push_back(tensor);
    }
    vector<shared_ptr<runtime::Tensor>> result_tensors;
    for (const shared_ptr<runtime::HostTensor>& tensor : tensors) {
        result_tensors.push_back(tensor);
    }
    return result_tensors;
}

bool runtime::interpreter::INTExecutable::evaluate_node(const std::shared_ptr<Node>& node,
                                                        const HostTensorVector& outputs,
                                                        const HostTensorVector& inputs) const {
    auto& map = runtime::interpreter::get_evaluators_map();
    auto it = map.find(node->get_type_info());
    bool res = false;
    if (it != map.end()) {
        res = it->second(node, outputs, inputs);
        if (!res) {
            throw ngraph_error(std::string("Running evaluate method for OP ") + node->get_type_info().name +
                               std::string(" failed!"));
        }
    } else {
        throw unsupported_op(std::string("Interpreter backend doesn't implement evaluate method for OP ") +
                             node->get_type_info().name);
    }
    return res;
}
