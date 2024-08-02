// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/function.hpp"

#include <cstring>

#include "openvino/core/shape_util.hpp"

namespace ov {
namespace reference {

void function(const std::shared_ptr<Model>& function, const ov::TensorVector& inputs, ov::TensorVector& outputs) {
    outputs.reserve(function->get_output_size());
    for (const auto& result : function->get_results()) {
        outputs.emplace_back(result->output(0));
    }
    function->evaluate(outputs, inputs);
}

void function(const std::shared_ptr<Model>& function,
              const ov::TensorVector& inputs,
              ov::TensorVector& outputs,
              const EvaluationContext& evaluation_context) {
    outputs.reserve(function->get_output_size());
    for (const auto& result : function->get_results()) {
        outputs.emplace_back(result->output(0));
    }
    function->evaluate(outputs, inputs, const_cast<EvaluationContext&>(evaluation_context));
}

}  // namespace reference
}  // namespace ov
