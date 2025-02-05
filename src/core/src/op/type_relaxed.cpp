// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/type_relaxed.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include "openvino/core/descriptor_tensor.hpp"

namespace ov {
namespace op {

TypeRelaxedBase::~TypeRelaxedBase() = default;

void TypeRelaxedBase::remember_input_data_types(Node& node, element::TypeVector& old_input_types) {
    // Remember all input data types
    for (size_t i = 0; i < node.get_input_size(); ++i) {
        old_input_types.push_back(node.get_input_element_type(i));
    }

    // Reset input data types to m_output_data_type.
    for (size_t i = 0; i < node.get_input_size(); ++i) {
        auto origin_input_type = get_origin_input_type(i);
        if (origin_input_type.is_static()) {
            ov::descriptor::set_tensor_type(node.get_input_tensor(i),
                                            origin_input_type,
                                            node.get_input_partial_shape(i));
        }
    }
}

void TypeRelaxedBase::restore_input_data_types(Node& node, const element::TypeVector& old_input_types) {
    // Restore original input data types
    for (size_t i = 0; i < node.get_input_size(); ++i) {
        ov::descriptor::set_tensor_type(node.get_input_tensor(i), old_input_types[i], node.get_input_partial_shape(i));
    }

    if (m_original_output_data_types.empty()) {
        m_original_output_data_types = element::TypeVector(node.get_output_size());
    }

    // Save inferred output types
    for (size_t i = 0; i < node.get_output_size(); ++i) {
        m_original_output_data_types[i] = node.get_output_element_type(i);
    }

    // Override (some) output types
    for (size_t i = 0; i < node.get_output_size(); ++i) {
        auto overridden_output_type = get_overridden_output_type(i);
        if (overridden_output_type.is_static()) {
            node.set_output_type(i, overridden_output_type, node.get_output_partial_shape(i));
        }
    }
}

TemporaryReplaceOutputType::TemporaryReplaceOutputType(Output<Node> output, element::Type tmp_type)
    : m_output(std::move(output)),
      orig_type(m_output.get_element_type()) {
    // save original element type in order to restore it in the destructor
    ov::descriptor::set_element_type(m_output.get_tensor(), tmp_type);
}

Output<Node> TemporaryReplaceOutputType::get() const {
    return m_output;
}

TemporaryReplaceOutputType::~TemporaryReplaceOutputType() {
    ov::descriptor::set_element_type(m_output.get_tensor(), orig_type);
}

namespace {
void convert_types(std::shared_ptr<v0::Parameter>& parameter,
                   std::shared_ptr<v0::Convert>& convert,
                   Output<Node>& output,
                   const element::Type& new_type) {
    parameter->set_element_type(output.get_element_type());
    parameter->set_partial_shape(output.get_shape());
    parameter->validate_and_infer_types();
    if (auto& bound = output.get_tensor().get_lower_value())
        parameter->get_output_tensor(0).set_lower_value(bound);
    if (auto& bound = output.get_tensor().get_upper_value())
        parameter->get_output_tensor(0).set_upper_value(bound);

    convert->set_destination_type(new_type);
    convert->validate_and_infer_types();

    ov::TensorVector lower_tensor = {ov::Tensor(new_type, output.get_shape())};
    ov::TensorVector upper_tensor = {ov::Tensor(new_type, output.get_shape())};
    bool lower_success = convert->evaluate_lower(lower_tensor);
    bool upper_success = convert->evaluate_upper(upper_tensor);
    auto& tensor = output.get_tensor();

    if (lower_success || upper_success) {
        ov::descriptor::set_element_type(tensor, new_type);
    }
    if (lower_success)
        tensor.set_lower_value(lower_tensor[0]);
    if (upper_success)
        tensor.set_upper_value(upper_tensor[0]);
    if (lower_success && upper_success) {
        if (memcmp(lower_tensor[0].data(), upper_tensor[0].data(), lower_tensor[0].get_byte_size()) == 0)
            tensor.set_upper_value(lower_tensor[0]);
    }
}

void reset_convert(std::shared_ptr<v0::Parameter> parameter,
                   std::shared_ptr<v0::Convert> convert,
                   const ov::Tensor& tensor,
                   const element::Type& new_type,
                   bool is_upper) {
    parameter->set_element_type(tensor.get_element_type());
    parameter->set_partial_shape(tensor.get_shape());
    parameter->validate_and_infer_types();
    convert->set_destination_type(new_type);
    convert->validate_and_infer_types();
    if (is_upper) {
        convert->get_input_tensor(0).set_upper_value(tensor);
    } else {
        convert->get_input_tensor(0).set_lower_value(tensor);
    }
}
}  // namespace

std::unordered_map<size_t, std::pair<ov::Tensor, ov::Tensor>> convert_input_types(OutputVector& inputs,
                                                                                  const element::TypeVector& types) {
    OPENVINO_ASSERT(inputs.size() >= types.size());
    std::shared_ptr<v0::Parameter> parameter = nullptr;
    std::shared_ptr<v0::Convert> convert = nullptr;
    std::unordered_map<size_t, std::pair<ov::Tensor, ov::Tensor>> original_inputs;  // input_idx -> {lower, upper}
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (i >= types.size())
            break;  // inputs with this idx and higher don't change type
        auto& input = inputs[i];
        const auto& fake_type = input.get_element_type();
        const auto& original_type = types[i];
        if (original_type == fake_type || original_type == element::dynamic)
            continue;  // this input type wasn't changed
        if (parameter == nullptr || convert == nullptr) {
            parameter = std::make_shared<op::v0::Parameter>(element::dynamic, PartialShape());
            convert = std::make_shared<ov::op::v0::Convert>(parameter, element::dynamic);
        }
        ov::op::convert_types(parameter, convert, input, original_type);
        original_inputs[i] = {parameter->get_output_tensor(0).get_lower_value(),
                              parameter->get_output_tensor(0).get_upper_value()};
    }
    return original_inputs;
}

ov::TensorVector get_output_tensors_of_original_type(const ov::TensorVector& fake_output_tensors,
                                                     const element::TypeVector& types) {
    TensorVector original_outputs(fake_output_tensors.size());
    for (size_t i = 0; i < original_outputs.size(); ++i) {
        const auto fake_type = fake_output_tensors[i].get_element_type();
        const auto original_type = types[i];
        if (fake_type == original_type) {
            original_outputs[i] = fake_output_tensors[i];
        } else {
            original_outputs[i] = ov::Tensor(original_type, fake_output_tensors[i].get_shape());
        }
    }
    return original_outputs;
}

void reset_input_types(const std::unordered_map<size_t, std::pair<ov::Tensor, ov::Tensor>>& original_input_vals,
                       OutputVector& inputs) {
    for (auto& item : original_input_vals) {
        if (!item.second.first && !item.second.second)
            continue;
        const auto& etype =
            item.second.first ? item.second.first.get_element_type() : item.second.second.get_element_type();
        auto& tensor = inputs[item.first].get_tensor();
        ov::descriptor::set_element_type(tensor, etype);
        if (item.second.first)
            tensor.set_lower_value(item.second.first);
        if (item.second.second)
            tensor.set_upper_value(item.second.second);
    }
}

bool convert_outputs_to_fake_type(ov::TensorVector& outputs, ov::TensorVector& original_outputs, bool is_upper) {
    OPENVINO_ASSERT(outputs.size() == original_outputs.size());
    std::shared_ptr<v0::Parameter> parameter = nullptr;
    std::shared_ptr<v0::Convert> convert = nullptr;
    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& fake_type = outputs[i].get_element_type();
        const auto& original_type = original_outputs[i].get_element_type();
        if (fake_type == original_type)
            continue;
        if (parameter == nullptr || convert == nullptr) {
            parameter = std::make_shared<op::v0::Parameter>(element::dynamic, PartialShape());
            convert = std::make_shared<ov::op::v0::Convert>(parameter, element::dynamic);
        }
        reset_convert(parameter, convert, original_outputs[i], fake_type, is_upper);
        TensorVector local_outputs = {outputs[i]};
        if (is_upper && !convert->evaluate_upper(local_outputs))
            return false;
        if (!is_upper && !convert->evaluate_lower(local_outputs))
            return false;
    }
    return true;
}

}  // namespace op
}  // namespace ov
