// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/if.hpp"

#include "evaluate_node.hpp"
#include "evaluates_map.hpp"
#include "openvino/core/shape_util.hpp"

namespace if_op {
bool call(ov::TensorVector& func_outputs,
          const ov::TensorVector& func_inputs,
          const std::shared_ptr<ov::Model>& function) {
    // map function params -> ov::Tensor
    std::unordered_map<ov::descriptor::Tensor*, ov::Tensor> tensor_map;
    size_t input_count = 0;
    for (const auto& param : function->get_parameters()) {
        for (size_t i = 0; i < param->get_output_size(); ++i) {
            ov::descriptor::Tensor* tensor = &param->output(i).get_tensor();
            tensor_map.insert({tensor, func_inputs[input_count++]});
        }
    }

    std::unordered_map<std::shared_ptr<ov::Node>, size_t> results_map;
    // map function outputs -> ov::Tensor
    for (size_t output_count = 0; output_count < function->get_results().size(); ++output_count) {
        auto output = function->get_results()[output_count];
        results_map[output] = output_count;
    }

    // for each ordered op in the graph
    for (const auto& op : function->get_ordered_ops()) {
        if (ov::op::util::is_parameter(op)) {
            continue;
        }

        // get op inputs from map
        std::vector<ov::Tensor> op_inputs;
        for (auto input : op->inputs()) {
            ov::descriptor::Tensor* tensor = &input.get_tensor();
            op_inputs.push_back(tensor_map.at(tensor));
        }

        // get op outputs from map or create
        std::vector<ov::Tensor> op_outputs;
        for (size_t i = 0; i < op->get_output_size(); ++i) {
            ov::descriptor::Tensor* tensor = &op->output(i).get_tensor();
            ov::Tensor host_tensor;
            auto it = tensor_map.find(tensor);
            if (ov::op::util::is_output(op)) {
                host_tensor = func_outputs[results_map[op]];
            } else if (it == tensor_map.end()) {
                host_tensor = ov::Tensor(op->output(i));
                tensor_map.insert({tensor, host_tensor});
            } else {
                host_tensor = it->second;
            }
            op_outputs.push_back(host_tensor);
        }
        op->validate_and_infer_types();
        if (!op->evaluate(op_outputs, op_inputs)) {
            const auto& evaluates_map = ov::runtime::interpreter::get_evaluators_map();
            auto it = evaluates_map.find(op->get_type_info());
            if (!it->second(op, op_outputs, op_inputs)) {
                return false;
            }
        }
    }
    return true;
}

void function(const std::shared_ptr<ov::Model>& function, const ov::TensorVector& inputs, ov::TensorVector& outputs) {
    const auto& parameters = function->get_parameters();
    const auto& parametersNumber = parameters.size();
    const auto& inputsNumber = inputs.size();
    OPENVINO_ASSERT(parametersNumber == inputsNumber,
                    "Got function (",
                    function->get_friendly_name(),
                    ") with ",
                    parametersNumber,
                    " parameters, but ",
                    inputsNumber,
                    " input blobs");

    for (const auto& parameter : parameters) {
        const auto& parameterIndex = function->get_parameter_index(parameter);
        const auto& parameterShape = parameter->get_shape();
        const auto& parameterType = parameter->get_element_type();
        const auto& parameterSize = ov::shape_size(parameterShape) * parameterType.size();

        const auto& input = inputs[parameterIndex];
        const auto& inputSize = input.get_byte_size();
        OPENVINO_ASSERT(parameterSize == inputSize,
                        "Got parameter (",
                        parameter->get_friendly_name(),
                        ") of size ",
                        parameterSize,
                        " bytes, but corresponding input with index ",
                        parameterIndex,
                        " has ",
                        inputSize,
                        " bytes");
    }

    outputs.reserve(function->get_output_size());
    for (const auto& result : function->get_results()) {
        outputs.emplace_back(result->output(0));
    }
    call(outputs, inputs, function);
}

void if_reference(const std::vector<std::shared_ptr<ov::Model>>& bodies,
                  const std::vector<ov::op::util::MultiSubGraphOp::MultiSubgraphOutputDescriptionVector>& out_descs,
                  const std::vector<ov::op::util::MultiSubGraphOp::MultiSubgraphInputDescriptionVector>& input_descs,
                  ov::TensorVector& out,
                  const ov::TensorVector& args) {
    OPENVINO_ASSERT(args.size() > 0, "If operation must have input condition value");

    auto condition_value = args[0].data<bool>()[0];
    auto branch_index = (condition_value) ? ov::op::v8::If::THEN_BODY_INDEX : ov::op::v8::If::ELSE_BODY_INDEX;
    ov::TensorVector inputs_to_body;
    ov::TensorVector outs_from_body;
    inputs_to_body.resize(input_descs[branch_index].size());
    auto inputs_size = args.size();
    auto output_size = out.size();
    for (const auto& input_desc : input_descs[branch_index]) {
        OPENVINO_ASSERT(inputs_size > input_desc->m_input_index,
                        "Incorrect associating! If has not input with id ",
                        input_desc->m_input_index);
        inputs_to_body[input_desc->m_body_parameter_index] = args[input_desc->m_input_index];
    }
    function(bodies[branch_index], inputs_to_body, outs_from_body);
    for (const auto& out_descr : out_descs[branch_index]) {
        OPENVINO_ASSERT(output_size > out_descr->m_output_index,
                        "Incorrect associating! If has not output with id ",
                        out_descr->m_output_index);
        auto res = outs_from_body[out_descr->m_body_value_index];

        res.copy_to(out[out_descr->m_output_index]);
    }
}
}  // namespace if_op

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v8::If>& op, ov::TensorVector& outputs, const ov::TensorVector& inputs) {
    std::vector<std::shared_ptr<ov::Model>> bodies;
    for (size_t i = 0; i < op->get_internal_subgraphs_size(); i++) {
        bodies.emplace_back(op->get_function(static_cast<int>(i)));
    }
    std::vector<ov::op::util::MultiSubGraphOp::MultiSubgraphInputDescriptionVector> in_descs;
    for (size_t i = 0; i < op->get_input_descriptions_size(); i++) {
        in_descs.emplace_back(op->get_input_descriptions(static_cast<int>(i)));
    }
    std::vector<ov::op::util::MultiSubGraphOp::MultiSubgraphOutputDescriptionVector> out_descs;
    for (size_t i = 0; i < op->get_output_descriptions_size(); i++) {
        out_descs.emplace_back(op->get_output_descriptions(static_cast<int>(i)));
    }
    try {
        ov::reference::if_reference(bodies, out_descs, in_descs, outputs, inputs);
    } catch (...) {
        if_op::if_reference(bodies, out_descs, in_descs, outputs, inputs);
    }
    return true;
}

template <>
bool evaluate_node<ov::op::v8::If>(std::shared_ptr<ov::Node> node,
                                   ov::TensorVector& outputs,
                                   const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v8::If>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v8::If>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v8::If>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v8::If>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v8::If>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v8::If>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v8::If>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v8::If>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v8::If>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v8::If>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v8::If>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v8::If>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v8::If>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v8::If>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v8::If>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v8::If>(node), outputs, inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}
