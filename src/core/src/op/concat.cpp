// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/concat.hpp"

#include <memory>

#include "bound_evaluate.hpp"
#include "concat_shape_inference.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/validation_util.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/reference/concat.hpp"
#include "validation_util.hpp"

using namespace std;
using namespace ngraph;

op::Concat::Concat(const OutputVector& args, int64_t axis) : Op(args), m_axis(axis) {
    constructor_validate_and_infer_types();
}

op::Concat::Concat(const NodeVector& args, int64_t axis) : Concat(as_output_vector(args), axis) {}

bool op::Concat::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Concat_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    return true;
}

void op::Concat::validate_and_infer_types() {
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

shared_ptr<Node> op::Concat::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Concat_clone_with_new_inputs);
    return make_shared<Concat>(new_args, m_axis);
}

OPENVINO_SUPPRESS_DEPRECATED_START
namespace {
bool evaluate_concat(const HostTensorVector& args, const HostTensorPtr& out, int64_t concatenation_axis) {
    std::vector<const char*> arg_bufs;
    std::vector<ov::Shape> arg_shapes;
    ov::Shape out_shape(args[0]->get_shape());
    out_shape[concatenation_axis] = 0;
    for (auto& input : args) {
        arg_bufs.push_back(input->get_data_ptr<char>());
        arg_shapes.push_back(input->get_shape());
        out_shape[concatenation_axis] += arg_shapes.back()[concatenation_axis];
    }
    out->set_shape(out_shape);
    ov::reference::concat(arg_bufs,
                          out->get_data_ptr<char>(),
                          arg_shapes,
                          out_shape,
                          concatenation_axis,
                          out->get_element_type().size());

    return true;
}
}  // namespace

bool op::Concat::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_Concat_evaluate);
    OPENVINO_ASSERT(!inputs.empty());
    OPENVINO_ASSERT(validate_host_tensor_vector(inputs, inputs.size()));
    OPENVINO_ASSERT(validate_host_tensor_vector(outputs, 1));
    auto concat_axis = get_axis() < 0 ? get_axis() + inputs[0]->get_shape().size() : get_axis();
    return evaluate_concat(inputs, outputs[0], concat_axis);
}
OPENVINO_SUPPRESS_DEPRECATED_END

bool op::Concat::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Concat_evaluate);
    OPENVINO_ASSERT(!inputs.empty());
    OPENVINO_ASSERT(outputs.size() == 1);

    auto concat_axis = ov::util::normalize(get_axis(), inputs.front().get_shape().size());

    std::vector<const char*> arg_bufs;
    std::vector<ov::Shape> arg_shapes;

    ov::Shape out_shape(inputs.front().get_shape());
    out_shape[concat_axis] = 0;
    for (auto& input : inputs) {
        arg_bufs.push_back(static_cast<const char*>(input.data()));
        arg_shapes.push_back(input.get_shape());
        out_shape[concat_axis] += arg_shapes.back()[concat_axis];
    }
    outputs.front().set_shape(out_shape);
    ov::reference::concat(arg_bufs,
                          static_cast<char*>(outputs.front().data()),
                          arg_shapes,
                          out_shape,
                          concat_axis,
                          outputs.front().get_element_type().size());

    return true;
}

bool op::Concat::has_evaluate() const {
    OV_OP_SCOPE(v0_Concat_has_evaluate);
    return true;
}

bool op::Concat::evaluate_lower(ov::TensorVector& output_values) const {
    return default_lower_bound_evaluator(this, output_values);
}

bool op::Concat::evaluate_upper(ov::TensorVector& output_values) const {
    return default_upper_bound_evaluator(this, output_values);
}

bool op::Concat::evaluate_label(TensorLabelVector& output_labels) const {
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
        idx_inputs.emplace_back(element::from<label_t>(), input.get_partial_shape().to_shape());
        std::copy_n(input_label.begin(), idx_inputs.back().get_size(), idx_inputs.back().data<ov::label_t>());
    }

    auto outputs = TensorVector{{element::from<label_t>(), get_output_partial_shape(0).to_shape()}};
    if (evaluate(outputs, idx_inputs)) {
        output_labels.front() =
            TensorLabel(outputs.front().data<label_t>(), outputs.front().data<label_t>() + outputs.front().get_size());
        return true;
    } else {
        return false;
    }
}
