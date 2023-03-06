// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "int_executable.hpp"

#include <cstring>
#include <limits>
#include <openvino/op/util/variable_context.hpp>

#include "evaluates_map.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/op_types.hpp"
#include "tensor_conversion_util.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

namespace {

class DynamicTensor : public ngraph::runtime::HostTensor {
private:
    ov::Tensor tensor;

public:
    DynamicTensor(const ov::element::Type& type) : ngraph::runtime::HostTensor(type, ov::PartialShape::dynamic()) {}

    ov::Tensor get_tensor() {
        return tensor;
    }

protected:
    void allocate_buffer() override {
        OPENVINO_ASSERT(get_partial_shape().is_static(),
                        "Attempt to allocate buffer for tensor with partial shape: ",
                        get_partial_shape());
        OPENVINO_ASSERT(get_element_type().is_static(),
                        "Attempt to allocate buffer for tensor with dynamic type: ",
                        get_element_type());
        m_buffer_size = m_descriptor->size();
        tensor = ov::Tensor(get_element_type(), get_partial_shape().get_shape());
        m_memory_pointer = tensor.data();
        m_aligned_buffer_pool = m_memory_pointer;
    }
};

inline ngraph::HostTensorPtr make_tmp_host_tensor(const ov::Tensor& t) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    if (!t) {
        return std::make_shared<DynamicTensor>(ov::element::dynamic);
    } else if (t.get_shape() == ov::Shape{0, std::numeric_limits<size_t>::max()}) {
        return std::make_shared<DynamicTensor>(t.get_element_type());
    } else {
        return std::make_shared<ngraph::runtime::HostTensor>(t.get_element_type(), t.get_shape(), t.data());
    }
    OPENVINO_SUPPRESS_DEPRECATED_END
}
inline ngraph::HostTensorVector create_tmp_tensors(const ov::TensorVector& tensors) {
    ngraph::HostTensorVector result;
    result.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        result.push_back(make_tmp_host_tensor(tensor));
    }
    return result;
}

inline void update_output_tensors(ov::TensorVector& output_values, const ngraph::HostTensorVector& outputs) {
    OPENVINO_ASSERT(output_values.size() == outputs.size());
    for (size_t i = 0; i < outputs.size(); i++) {
        if (auto dyn_output = std::dynamic_pointer_cast<DynamicTensor>(outputs[i])) {
            output_values[i] = dyn_output->get_tensor();
        }
    }
}
}  // namespace

class TemporaryOverrideOutputs {
    std::shared_ptr<ov::Node> node;
    std::vector<ov::PartialShape> orig_shapes;

public:
    TemporaryOverrideOutputs(std::shared_ptr<ov::Node> node, const std::vector<ov::Tensor>& args) : node(node) {
        for (size_t i = 0; i < args.size(); ++i) {
            auto output = node->get_input_source_output(i);
            orig_shapes.push_back(output.get_partial_shape());
            output.get_tensor().set_partial_shape(args[i].get_shape());
        }
    }

    ~TemporaryOverrideOutputs() {
        for (size_t i = 0; i < orig_shapes.size(); ++i) {
            auto output = node->get_input_source_output(i);
            output.get_tensor().set_partial_shape(orig_shapes[i]);
        }
    }
};

ov::runtime::interpreter::INTExecutable::INTExecutable(const std::shared_ptr<ov::Model>& model) : m_is_compiled{true} {
    m_model = model->clone();
    for (auto node : m_model->get_ordered_ops()) {
        m_nodes.push_back(node);
    }
    set_parameters_and_results(*m_model);
}

bool ov::runtime::interpreter::INTExecutable::call(std::vector<ov::Tensor>& outputs,
                                                   const std::vector<ov::Tensor>& inputs) {
    // map function params -> HostTensor
    std::unordered_map<std::shared_ptr<ov::descriptor::Tensor>, ov::Tensor> tensor_map;
    size_t input_count = 0;
    for (const auto& param : get_parameters()) {
        for (size_t i = 0; i < param->get_output_size(); ++i) {
            auto tensor = param->output(i).get_tensor_ptr();
            tensor_map.insert({tensor, inputs[input_count++]});
        }
    }

    std::unordered_map<std::shared_ptr<ov::descriptor::Tensor>, size_t> results_map;
    // map function outputs -> HostTensor
    for (size_t output_count = 0; output_count < get_results().size(); ++output_count) {
        auto output = get_results()[output_count]->output(0).get_tensor_ptr();
        if (!results_map.count(output))
            results_map.emplace(output, output_count);
    }

    EvaluationContext eval_context;
    ov::op::util::VariableContext variable_context;
    eval_context.emplace("VariableContext", variable_context);

    // for each ordered op in the graph
    for (const auto& op : m_nodes) {
        if (std::dynamic_pointer_cast<ov::op::v0::Parameter>(op)) {
            continue;
        }

        // get op inputs from map
        std::vector<ov::Tensor> op_inputs;
        for (auto input : op->inputs()) {
            auto tensor = input.get_tensor_ptr();
            op_inputs.push_back(tensor_map.at(tensor));
        }

        TemporaryOverrideOutputs overrider(op, op_inputs);
        OutputVector output_ports;
        for (size_t i = 0; i < op->inputs().size(); ++i) {
            output_ports.push_back(op->get_input_source_output(i));
        }
        auto cloned_node = op->clone_with_new_inputs(output_ports);

        // get op outputs from map or create
        std::vector<ov::Tensor> op_outputs;
        for (size_t i = 0; i < op->get_output_size(); ++i) {
            auto tensor = op->output(i).get_tensor_ptr();
            ov::Tensor host_tensor;
            auto it = tensor_map.find(tensor);
            auto output = cloned_node->output(i);
            if (op::util::is_output(op) || it == tensor_map.end() || !it->second) {
                host_tensor = ov::Tensor(output.get_element_type(),
                                         output.get_partial_shape().is_dynamic()
                                             ? ov::Shape{0, std::numeric_limits<size_t>::max()}
                                             : output.get_shape());
            } else {
                host_tensor = it->second;
            }
            op_outputs.push_back(host_tensor);
        }

        if (auto var_extension = std::dynamic_pointer_cast<ov::op::util::VariableExtension>(cloned_node)) {
            auto variable = var_extension->get_variable();
            if (!variable_context.get_variable_value(variable)) {
                auto h_tensor = ov::Tensor(cloned_node->get_input_element_type(0), cloned_node->get_input_shape(0));
                // h_tensor->write(h_tensor->get_data_ptr(), h_tensor->get_size_in_bytes());
                const auto tensor_input = make_tmp_host_tensor(h_tensor);
                variable_context.set_variable_value(variable,
                                                    std::make_shared<ov::op::util::VariableValue>(tensor_input));
            }
        }

        // Call evaluate for cloned_node with static shapes
        if (!cloned_node->evaluate(op_outputs, op_inputs, eval_context)) {
            evaluate_node(cloned_node, op_outputs, op_inputs);
        }
        // Update tensors in tensor map
        for (size_t i = 0; i < op->get_output_size(); ++i) {
            auto tensor = op->output(i).get_tensor_ptr();
            tensor_map.insert({tensor, op_outputs[i]});
            if (op::util::is_output(op)) {
                auto& output = outputs[results_map[tensor]];
                if (!output || output.get_shape() != op_outputs[i].get_shape()) {
                    outputs[results_map[tensor]] = op_outputs[i];
                } else {
                    op_outputs[i].copy_to(output);
                }
            }
        }
    }

    return true;
}

std::shared_ptr<ov::op::v0::Parameter> ov::runtime::interpreter::INTExecutable::get_parameter(size_t index) const {
    const ParameterVector& parameters = get_parameters();
    NGRAPH_CHECK(index < parameters.size(), "create_tensor for input out of bounds");
    return parameters[index];
}

std::shared_ptr<ov::op::v0::Result> ov::runtime::interpreter::INTExecutable::get_result(size_t index) const {
    const ResultVector& results = get_results();
    NGRAPH_CHECK(index < results.size(), "create_tensor for input out of bounds");
    return results[index];
}
ov::Tensor ov::runtime::interpreter::INTExecutable::create_input_tensor(size_t input_index) {
    std::shared_ptr<op::v0::Parameter> parameter = get_parameter(input_index);
    return ov::Tensor(parameter->get_element_type(), parameter->get_shape());
}

ov::Tensor ov::runtime::interpreter::INTExecutable::create_output_tensor(size_t output_index) {
    std::shared_ptr<op::v0::Result> result = get_result(output_index);
    return ov::Tensor(result->get_element_type(), result->get_shape());
}

std::vector<ov::Tensor> ov::runtime::interpreter::INTExecutable::create_input_tensor(size_t input_index,
                                                                                     size_t pipeline_depth) {
    std::vector<ov::Tensor> tensors;
    std::shared_ptr<op::v0::Parameter> parameter = get_parameter(input_index);
    for (size_t i = 0; i < pipeline_depth; i++) {
        ov::Tensor tensor;
        auto t = ov::Tensor(parameter->get_element_type(), parameter->get_shape());
        tensors.push_back(t);
    }
    return tensors;
}

std::vector<ov::Tensor> ov::runtime::interpreter::INTExecutable::create_output_tensor(size_t output_index,
                                                                                      size_t pipeline_depth) {
    std::vector<ov::Tensor> tensors;
    std::shared_ptr<op::v0::Result> result = get_result(output_index);
    for (size_t i = 0; i < pipeline_depth; i++) {
        ov::Tensor tensor;
        auto t = ov::Tensor(result->get_element_type(), result->get_shape());
        tensors.push_back(t);
    }
    return tensors;
}

bool ov::runtime::interpreter::INTExecutable::evaluate_node(const std::shared_ptr<Node>& node,
                                                            ov::TensorVector& outputs,
                                                            const ov::TensorVector& inputs) const {
    auto& map = ngraph::runtime::interpreter::get_evaluators_map();
    auto it = map.find(node->get_type_info());
    bool res = false;
    const auto tensor_inputs = create_tmp_tensors(inputs);
    auto tensor_outputs = create_tmp_tensors(outputs);
    if (it != map.end()) {
        res = it->second(node, tensor_outputs, tensor_inputs);
        if (!res) {
            throw ngraph::ngraph_error(std::string("Running evaluate method for OP ") + node->get_type_info().name +
                                       std::string(" failed!"));
        }
        update_output_tensors(outputs, tensor_outputs);
    } else {
        throw ngraph::unsupported_op(std::string("Interpreter backend doesn't implement evaluate method for OP ") +
                                     node->get_type_info().name);
    }
    return res;
}
