// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/util/op_types.hpp"

namespace if_op {
bool call(const ov::HostTensorVector& func_outputs,
          const ov::HostTensorVector& func_inputs,
          const std::shared_ptr<ngraph::Function>& function) {
    // map function params -> HostTensor
    std::unordered_map<ov::descriptor::Tensor*, std::shared_ptr<ov::HostTensor>> tensor_map;
    size_t input_count = 0;
    for (const auto& param : function->get_parameters()) {
        for (size_t i = 0; i < param->get_output_size(); ++i) {
            ov::descriptor::Tensor* tensor = &param->output(i).get_tensor();
            tensor_map.insert({tensor, func_inputs[input_count++]});
        }
    }

    std::unordered_map<std::shared_ptr<ngraph::Node>, size_t> results_map;
    // map function outputs -> HostTensor
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
        std::vector<std::shared_ptr<ov::HostTensor>> op_inputs;
        for (auto input : op->inputs()) {
            ov::descriptor::Tensor* tensor = &input.get_tensor();
            op_inputs.push_back(tensor_map.at(tensor));
        }

        // get op outputs from map or create
        std::vector<std::shared_ptr<ov::HostTensor>> op_outputs;
        for (size_t i = 0; i < op->get_output_size(); ++i) {
            ov::descriptor::Tensor* tensor = &op->output(i).get_tensor();
            std::shared_ptr<ov::HostTensor> host_tensor;
            auto it = tensor_map.find(tensor);
            if (ov::op::util::is_output(op)) {
                host_tensor = func_outputs[results_map[op]];
            } else if (it == tensor_map.end()) {
                host_tensor = std::make_shared<ov::HostTensor>(op->output(i));
                tensor_map.insert({tensor, host_tensor});
            } else {
                host_tensor = it->second;
            }
            op_outputs.push_back(host_tensor);
        }
        op->validate_and_infer_types();
        OPENVINO_SUPPRESS_DEPRECATED_START
        if (!op->evaluate(op_outputs, op_inputs)) {
            const auto& evaluates_map = ngraph::runtime::interpreter::get_evaluators_map();
            auto it = evaluates_map.find(op->get_type_info());
            if (!it->second(op, op_outputs, op_inputs)) {
                return false;
            }
        }
        OPENVINO_SUPPRESS_DEPRECATED_END
    }
    return true;
}

void function(const std::shared_ptr<ngraph::Function>& function,
              const ov::HostTensorVector& inputs,
              ov::HostTensorVector& outputs) {
    const auto& parameters = function->get_parameters();
    const auto& parametersNumber = parameters.size();
    const auto& inputsNumber = inputs.size();
    NGRAPH_CHECK(parametersNumber == inputsNumber,
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
        const auto& parameterSize = shape_size(parameterShape) * parameterType.size();

        const auto& input = inputs[parameterIndex];
        const auto& inputSize = input->get_size_in_bytes();
        NGRAPH_CHECK(parameterSize == inputSize,
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

    const auto& results = function->get_results();
    outputs.reserve(results.size());
    for (size_t i = 0; i < results.size(); ++i) {
        outputs.push_back(std::make_shared<ov::HostTensor>());
    }
    call(outputs, inputs, function);
}

void if_reference(const std::vector<std::shared_ptr<ngraph::Function>>& bodies,
                  const std::vector<ngraph::op::util::MultiSubgraphOutputDescriptionVector>& out_descs,
                  const std::vector<ngraph::op::util::MultiSubgraphInputDescriptionVector>& input_descs,
                  const ov::HostTensorVector& out,
                  const ov::HostTensorVector& args) {
    NGRAPH_CHECK(args.size() > 0, "If operation must have input condition value");

    auto condition_value = args[0]->get_data_ptr<bool>()[0];
    auto branch_index = (condition_value) ? ov::op::v8::If::THEN_BODY_INDEX : ov::op::v8::If::ELSE_BODY_INDEX;
    ov::HostTensorVector inputs_to_body;
    ov::HostTensorVector outs_from_body;
    inputs_to_body.resize(input_descs[branch_index].size());
    auto inputs_size = args.size();
    auto output_size = out.size();
    for (const auto& input_desc : input_descs[branch_index]) {
        NGRAPH_CHECK(inputs_size > input_desc->m_input_index,
                     "Incorrect associating! If has not input with id ",
                     input_desc->m_input_index);
        inputs_to_body[input_desc->m_body_parameter_index] = args[input_desc->m_input_index];
    }
    function(bodies[branch_index], inputs_to_body, outs_from_body);
    for (const auto& out_descr : out_descs[branch_index]) {
        NGRAPH_CHECK(output_size > out_descr->m_output_index,
                     "Incorrect associating! If has not output with id ",
                     out_descr->m_output_index);
        auto res = outs_from_body[out_descr->m_body_value_index];
        out[out_descr->m_output_index]->set_shape(res->get_shape());
        out[out_descr->m_output_index]->write(res->get_data_ptr(), res->get_size_in_bytes());
    }
}
}  // namespace if_op

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v8::If>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    std::vector<std::shared_ptr<ngraph::Function>> bodies;
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
        ngraph::runtime::reference::if_reference(bodies, out_descs, in_descs, outputs, inputs);
    } catch (...) {
        if_op::if_reference(bodies, out_descs, in_descs, outputs, inputs);
    }
    return true;
}
