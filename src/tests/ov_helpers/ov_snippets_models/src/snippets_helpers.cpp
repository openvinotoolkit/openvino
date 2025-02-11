// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets_helpers.hpp"
#include "snippets/op/subgraph.hpp"

namespace ov {
namespace test {
namespace snippets {

void SnippetsFunctionBase::validate_params_shape(const std::vector<PartialShape>& input_shapes,
                                                 const ov::ParameterVector& params) {
    OPENVINO_ASSERT(params.size() == input_shapes.size(),
                    "Passed input shapes and produced function are inconsistent: number of params mismatch. Expected: ",
                    input_shapes.size(), ", actual: ", params.size());
    for (size_t i = 0; i < input_shapes.size(); i++) {
        const auto& cur_shape = params[i]->get_partial_shape();
        OPENVINO_ASSERT(input_shapes[i] == cur_shape,
                        "Passed input shapes (", input_shapes[i],
                        ") and produced function shape(", cur_shape,
                        ") are inconsistent.");
    }
}

void SnippetsFunctionBase::validate_function(const std::shared_ptr<Model> &f) const {
    OPENVINO_ASSERT(f != nullptr, "The test requires Model to be defined");
    const auto &params = f->get_parameters();
    validate_params_shape(input_shapes, params);
}

SnippetsFunctionCustomizable::SnippetsFunctionCustomizable(const std::vector<PartialShape>& inputShapes,
                                                           const std::vector<std::shared_ptr<Node>>& customOps,
                                                           const std::vector<size_t>&& customOpsNumInputs)
        : SnippetsFunctionBase(inputShapes), custom_ops{customOps}, custom_ops_num_inputs{customOpsNumInputs} {
    OPENVINO_ASSERT(custom_ops_num_inputs.size() == custom_ops.size(), "Got inconsistent numbers of custom ops and custom ops inputs");
    // We need to set dummy inputs to increase input arguments count,
    // so clone_with_new_inputs() could pass without errors inside initOriginal() and initReference().
    ResetCustomOpsInputs();
}

void SnippetsFunctionCustomizable::ResetCustomOpsInputs() {
    auto dummy_input = std::make_shared<ov::op::v0::Parameter>(precision, Shape{});
    for (size_t i = 0; i < custom_ops.size(); i++) {
        const NodeVector inputs(custom_ops_num_inputs[i], dummy_input);
        custom_ops[i]->set_arguments(inputs);
    }
}

}  // namespace snippets
}  // namespace test
}  // namespace ov