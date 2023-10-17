// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/concat.hpp"

#include "bound_evaluate.hpp"
#include "concat_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/reference/concat.hpp"

namespace ov {
namespace op {
namespace v0 {

Concat::Concat(const OutputVector& args, int64_t axis) : Op(args), m_axis(axis) {
    constructor_validate_and_infer_types();
}

Concat::Concat(const NodeVector& args, int64_t axis) : Concat(as_output_vector(args), axis) {}

bool Concat::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Concat_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    return true;
}

void Concat::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Concat_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() >= 1, "At least one argument required.");

    element::Type inputs_et{element::dynamic};
    auto input_shapes = std::vector<PartialShape>();

    for (size_t i = 0; i < get_input_size(); ++i) {
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(inputs_et, inputs_et, get_input_element_type(i)),
                              "Argument element types are inconsistent.");
        const auto& input_shape = get_input_partial_shape(i);
        const auto& input_rank = input_shape.rank();

        if (input_rank.is_static() && (get_concatenation_axis() < 0)) {
            set_concatenation_axis(get_axis() < 0 ? get_axis() + input_rank.get_length() : get_axis());
        }

        const auto concat_axis = get_concatenation_axis();

        NODE_VALIDATION_CHECK(this,
                              input_shape.is_dynamic() || (0 <= concat_axis && concat_axis < input_rank.get_length()),
                              "Concatenation axis (",
                              concat_axis,
                              ") is out of bounds [",
                              -input_rank.get_length(),
                              ", ",
                              input_rank.get_length() - 1,
                              "] for ",
                              "argument ",
                              i,
                              ", which has shape ",
                              input_shape,
                              ".");

        input_shapes.push_back(input_shape);
    }

    const auto output_shapes = shape_infer(this, input_shapes);
    set_output_type(0, inputs_et, output_shapes.front());
}

std::shared_ptr<Node> Concat::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Concat_clone_with_new_inputs);
    return std::make_shared<Concat>(new_args, m_axis);
}

bool Concat::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Concat_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    const auto inputs_count = inputs.size();
    std::vector<const char*> arg_bufs(inputs_count);
    std::vector<Shape> arg_shapes;
    std::vector<PartialShape> input_shapes;
    arg_shapes.reserve(inputs_count);
    input_shapes.reserve(inputs_count);

    auto arg_buf = arg_bufs.begin();
    for (auto& input : inputs) {
        *arg_buf = static_cast<const char*>(input.data());
        ++arg_buf;
        const auto& input_shape = input.get_shape();
        arg_shapes.emplace_back(input_shape);
        input_shapes.emplace_back(input_shape);
    }

    const auto& out_shape = shape_infer(this, input_shapes).front().to_shape();
    outputs.front().set_shape(out_shape);
    reference::concat(arg_bufs,
                      static_cast<char*>(outputs.front().data()),
                      arg_shapes,
                      out_shape,
                      ov::util::normalize(get_axis(), out_shape.size()),
                      outputs.front().get_element_type().size());

    return true;
}

bool Concat::has_evaluate() const {
    OV_OP_SCOPE(v0_Concat_has_evaluate);
    return true;
}

bool Concat::evaluate_lower(TensorVector& output_values) const {
    return default_lower_bound_evaluator(this, output_values);
}

bool Concat::evaluate_upper(TensorVector& output_values) const {
    return default_upper_bound_evaluator(this, output_values);
}

bool Concat::evaluate_label(TensorLabelVector& output_labels) const {
    const auto& inputs = input_values();
    if (std::all_of(inputs.cbegin(), inputs.cend(), [](const Output<Node>& out) {
            const auto& labels = out.get_tensor().get_value_label();
            OPENVINO_SUPPRESS_DEPRECATED_START
            return has_no_labels(labels);
            OPENVINO_SUPPRESS_DEPRECATED_END
        })) {
        return false;
    }

    TensorVector idx_inputs;
    idx_inputs.reserve(inputs.size());
    for (const auto& input : inputs) {
        auto input_label = input.get_tensor().get_value_label();
        if (input_label.empty()) {
            const auto& shape = input.get_partial_shape();
            // sanity check. at this point value propagation was successful
            OPENVINO_ASSERT(shape.is_static());
            const auto& num_elements = shape_size(shape.to_shape());
            input_label.resize(num_elements, no_label);
        }
        idx_inputs.emplace_back(element::from<label_t>(), input.get_shape());
        std::copy_n(input_label.begin(), idx_inputs.back().get_size(), idx_inputs.back().data<ov::label_t>());
    }

    auto outputs = TensorVector{{element::from<label_t>(), get_output_shape(0)}};
    if (evaluate(outputs, idx_inputs)) {
        output_labels.front() =
            TensorLabel(outputs.front().data<label_t>(), outputs.front().data<label_t>() + outputs.front().get_size());
        return true;
    } else {
        return false;
    }
}
}  // namespace v0
}  // namespace op
}  // namespace ov
