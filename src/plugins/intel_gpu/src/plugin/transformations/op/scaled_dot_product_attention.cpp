// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/scaled_dot_product_attention.hpp"
#include "scaled_dot_product_attention_shape_inference.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

ScaledDotProductAttention::ScaledDotProductAttention(
                    const ov::Output<Node>& Q,
                    const ov::Output<Node>& K,
                    const ov::Output<Node>& V,
                    const ov::Output<Node>& scale,
                    const ov::Output<Node>& attn_mask)
    : Op({Q, K, V, scale, attn_mask}) {
    validate_and_infer_types();
}

ScaledDotProductAttention::ScaledDotProductAttention(
                    const ov::Output<Node>& Q,
                    const ov::Output<Node>& K,
                    const ov::Output<Node>& V,
                    const ov::Output<Node>& scale)
    : Op({Q, K, V, scale}) {
    validate_and_infer_types();
}

void ScaledDotProductAttention::validate_and_infer_types() {
    const auto& input_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
        input_size == 4 || input_size == 5,
        "Number of inputs is incorrect. Current value is: ",
        input_size,
        ", expected 4 or 5.");

    auto out_type = get_input_element_type(0);
    for (size_t i = 1; i < 3; i++) {
        const auto& element_type = get_input_element_type(i);
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(out_type, out_type, element_type),
                              "Mixed input types are not supported.");
    }
    NODE_VALIDATION_CHECK(this,
                          out_type.is_real() || out_type.is_dynamic(),
                          "The element type of the input tensor must be a floating-point.");

    ov::op::v13::ScaledDotProductAttention op;
    std::vector<ov::PartialShape> input_shapes = {
        get_input_partial_shape(0),
        get_input_partial_shape(1),
        get_input_partial_shape(2),
    };

    const auto& output_shapes = ov::op::v13::shape_infer(&op, input_shapes);
    set_output_type(0, out_type, output_shapes[0]);
}

bool ScaledDotProductAttention::visit_attributes(ov::AttributeVisitor &visitor) {
    return true;
}

std::shared_ptr<ov::Node> ScaledDotProductAttention::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    switch (new_args.size()) {
    case 4:
        return std::make_shared<ScaledDotProductAttention>(
            new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
    case 5:
        return std::make_shared<ScaledDotProductAttention>(
            new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), new_args.at(4));
    default:
        OPENVINO_THROW("Unable to clone ScaledDotProductAttention ",
                       this->get_friendly_name(),
                       " Incorrect number of inputs. Expected: 4 or 5. Actual: ",
                       new_args.size());
    }
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
