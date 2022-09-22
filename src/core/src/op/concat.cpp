// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/concat.hpp"

#include <memory>
#include <ngraph/validation_util.hpp>

#include "concat_shape_inference.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/runtime/reference/concat.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(ov::op::v0::Concat);

op::Concat::Concat(const OutputVector& args, int64_t axis) : Op(args), m_axis(axis) {
    constructor_validate_and_infer_types();
}

op::Concat::Concat(const NodeVector& args, int64_t axis) : Concat(as_output_vector(args), axis) {}

bool op::Concat::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v0_Concat_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    return true;
}

void op::Concat::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v0_Concat_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() >= 1, "At least one argument required.");

    ov::PartialShape inputs_shape_scheme{ov::PartialShape::dynamic()};
    element::Type inputs_et{element::dynamic};

    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};
    std::vector<ov::PartialShape> input_shapes;
    for (size_t i = 0; i < get_input_size(); i++) {
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(inputs_et, inputs_et, get_input_element_type(i)),
                              "Argument element types are inconsistent.");
        input_shapes.push_back(get_input_partial_shape(i));
    }
    shape_infer(this, input_shapes, output_shapes);
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

shared_ptr<Node> op::Concat::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v0_Concat_clone_with_new_inputs);
    // TODO(amprocte): Should we check the new_args count here?
    return make_shared<Concat>(new_args, m_axis);
}

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
    runtime::reference::concat(arg_bufs,
                               out->get_data_ptr<char>(),
                               arg_shapes,
                               out_shape,
                               concatenation_axis,
                               out->get_element_type().size());

    return true;
}
}  // namespace

bool op::Concat::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    NGRAPH_OP_SCOPE(v0_Concat_evaluate);
    NGRAPH_CHECK(!inputs.empty());
    NGRAPH_CHECK(validate_host_tensor_vector(inputs, inputs.size()));
    NGRAPH_CHECK(validate_host_tensor_vector(outputs, 1));
    auto concat_axis = get_axis() < 0 ? get_axis() + inputs[0]->get_shape().size() : get_axis();
    return evaluate_concat(inputs, outputs[0], concat_axis);
}

bool op::Concat::has_evaluate() const {
    NGRAPH_OP_SCOPE(v0_Concat_has_evaluate);
    return true;
}

bool op::Concat::evaluate_lower(const HostTensorVector& output_values) const {
    return default_lower_bound_evaluator(this, output_values);
}

bool op::Concat::evaluate_upper(const HostTensorVector& output_values) const {
    return default_upper_bound_evaluator(this, output_values);
}

bool op::Concat::evaluate_label(TensorLabelVector& output_labels) const {
    const auto& inputs = input_values();
    bool has_labeled_input = std::any_of(inputs.begin(), inputs.end(), [](const Output<Node>& out) {
        const auto& labels = out.get_tensor().get_value_label();
        return !labels.empty() && std::any_of(labels.begin(), labels.end(), [](const size_t& l) {
            return l > 0;
        });
    });
    if (!has_labeled_input)
        return false;

    HostTensorVector idx_inputs;
    idx_inputs.reserve(inputs.size());
    for (const auto& input : inputs) {
        auto input_label = input.get_tensor().get_value_label();
        if (input_label.empty()) {
            const auto& shape = input.get_partial_shape();
            // sanity check. at this point value propagation was successful
            NGRAPH_CHECK(shape.is_static());
            const auto& num_elements = shape_size(shape.to_shape());
            input_label = TensorLabel(num_elements, 0);
        }
        const auto& constant = Constant::create(element::u64, input.get_shape(), input_label);
        idx_inputs.push_back(std::make_shared<HostTensor>(constant));
    }

    const auto& output_tensor = std::make_shared<HostTensor>(element::u64, get_output_shape(0));
    evaluate({output_tensor}, idx_inputs);
    const auto& output_idxs = std::make_shared<Constant>(output_tensor)->cast_vector<size_t>();
    output_labels[0] = output_idxs;
    return true;
}