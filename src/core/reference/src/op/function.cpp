// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/function.hpp"

#include <cstring>

#include "openvino/core/shape_util.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/convert.hpp"

namespace ov {
namespace reference {

void function(const std::shared_ptr<Model>& function, const ov::TensorVector& inputs, ov::TensorVector& outputs) {
    outputs.reserve(function->get_output_size());
    for (const auto& result : function->get_results()) {
        auto result_type = result->output(0).get_element_type();
        // For unsupported types (f16, bf16), create tensors with f32 type to avoid type conversion errors
        // This handles the case where If/Loop subgraphs have f16 constants that need evaluation
        if (result_type == element::f16 || result_type == element::bf16) {
            outputs.emplace_back(element::f32, result->output(0).get_shape());
        } else {
            outputs.emplace_back(result->output(0));
        }
    }
    function->evaluate(outputs, inputs);

    // Convert back to original type if needed
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto expected_type = function->get_results()[i]->output(0).get_element_type();
        if (outputs[i].get_element_type() != expected_type) {
            ov::Tensor converted_output(expected_type, outputs[i].get_shape());
            ov::TensorVector convert_outputs = {converted_output};
            ov::op::v0::Convert().evaluate(convert_outputs, ov::TensorVector{outputs[i]});
            outputs[i] = converted_output;
        }
    }
}

void function(const std::shared_ptr<Model>& function,
              const ov::TensorVector& inputs,
              ov::TensorVector& outputs,
              const EvaluationContext& evaluation_context) {
    outputs.reserve(function->get_output_size());
    for (const auto& result : function->get_results()) {
        auto result_type = result->output(0).get_element_type();
        // For unsupported types (f16, bf16), create tensors with f32 type to avoid type conversion errors
        // This handles the case where If/Loop subgraphs have f16 constants that need evaluation
        if (result_type == element::f16 || result_type == element::bf16) {
            outputs.emplace_back(element::f32, result->output(0).get_shape());
        } else {
            outputs.emplace_back(result->output(0));
        }
    }
    function->evaluate(outputs, inputs, const_cast<EvaluationContext&>(evaluation_context));

    // Convert back to original type if needed
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto expected_type = function->get_results()[i]->output(0).get_element_type();
        if (outputs[i].get_element_type() != expected_type) {
            ov::Tensor converted_output(expected_type, outputs[i].get_shape());
            ov::TensorVector convert_outputs = {converted_output};
            ov::op::v0::Convert().evaluate(convert_outputs, ov::TensorVector{outputs[i]});
            outputs[i] = converted_output;
        }
    }
}

}  // namespace reference
}  // namespace ov
