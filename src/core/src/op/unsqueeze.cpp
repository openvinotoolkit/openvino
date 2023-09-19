// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/unsqueeze.hpp"

#include <cstddef>
#include <functional>
#include <set>

#include "bound_evaluate.hpp"
#include "element_visitor.hpp"
#include "itt.hpp"
#include "ngraph/validation_util.hpp"
#include "openvino/reference/copy.hpp"
#include "unsqueeze_shape_inference.hpp"

using namespace std;

ov::op::v0::Unsqueeze::Unsqueeze(const ov::Output<ov::Node>& data, const ov::Output<ov::Node>& axes)
    : Op({data, axes}) {
    constructor_validate_and_infer_types();
}

void ov::op::v0::Unsqueeze::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Unsqueeze_validate_and_infer_types);

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto input_shapes = get_node_input_partial_shapes(*this);
    OPENVINO_SUPPRESS_DEPRECATED_END
    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

bool ov::op::v0::Unsqueeze::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Unsqueeze_visit_attributes);
    return true;
}

shared_ptr<ov::Node> ov::op::v0::Unsqueeze::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Unsqueeze_clone_with_new_inputs);
    if (new_args.size() != 2) {
        OPENVINO_THROW("Incorrect number of new arguments");
    }
    return make_shared<Unsqueeze>(new_args.at(0), new_args.at(1));
}

namespace ov {
namespace op {
namespace unsqueeze {
// The evaluate cannot use shape_infer for output shape calculation as shape inference accepts
// repeated axis and evaluate not. When shape inference will changed to be compatible with `numpy` then
// evaluate and inference can use same function to calculate output shape. TODO for next version for this operator.
namespace {
bool evaluate_unsqueeze(const Node* node,
                        const HostTensorPtr& arg0,
                        const HostTensorPtr& arg1,
                        const HostTensorPtr& out) {
    auto element_type = arg0->get_element_type();
    out->set_element_type(element_type);

    const auto& axes_shape = arg1->get_shape();
    ov::op::v0::check_unsqueeze_axes_rank(node, Rank(axes_shape.size()));

    const auto& data_shape = arg0->get_shape();
    const auto out_rank = static_cast<int64_t>(data_shape.size() + shape_size(axes_shape));

    // Get axes and normalize
    OPENVINO_SUPPRESS_DEPRECATED_START
    auto axes = read_index_vector(arg1);
    normalize_axes(node, out_rank, axes);
    OPENVINO_SUPPRESS_DEPRECATED_END

    // Sort in increasing order
    std::set<int64_t> axes_set(axes.begin(), axes.end());
    NGRAPH_CHECK(axes.size() == axes_set.size(), "Axes has duplicate axis.");

    auto out_shape = data_shape;
    for (int64_t axis : axes_set) {
        out_shape.insert(out_shape.begin() + axis, 1);
    }
    out->set_shape(out_shape);

    ov::reference::copy(static_cast<const char*>(arg0->get_data_ptr()),
                        static_cast<char*>(out->get_data_ptr()),
                        out->get_size_in_bytes());
    return true;
}
}  // namespace
}  // namespace unsqueeze
}  // namespace op
}  // namespace ov

bool ov::op::v0::Unsqueeze::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_Unsqueeze_evaluate);
    OPENVINO_SUPPRESS_DEPRECATED_START
    NGRAPH_CHECK(ngraph::validate_host_tensor_vector(inputs, 2));
    NGRAPH_CHECK(ngraph::validate_host_tensor_vector(outputs, 1));
    OPENVINO_SUPPRESS_DEPRECATED_END
    return unsqueeze::evaluate_unsqueeze(this, inputs[0], inputs[1], outputs[0]);
}

bool ov::op::v0::Unsqueeze::has_evaluate() const {
    OV_OP_SCOPE(v0_Unsqueeze_has_evaluate);
    return true;
}

bool ov::op::v0::Unsqueeze::evaluate_lower(ov::TensorVector& output_values) const {
    return get_input_tensor(1).has_and_set_bound() && default_lower_bound_evaluator(this, output_values);
}

bool ov::op::v0::Unsqueeze::evaluate_upper(ov::TensorVector& output_values) const {
    return get_input_tensor(1).has_and_set_bound() && default_upper_bound_evaluator(this, output_values);
}

bool ov::op::v0::Unsqueeze::evaluate_label(TensorLabelVector& output_labels) const {
    if (!get_input_tensor(1).has_and_set_bound())
        return false;
    OPENVINO_SUPPRESS_DEPRECATED_START
    return default_label_evaluator(this, output_labels);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

bool ov::op::v0::Unsqueeze::constant_fold(OutputVector& output_values, const OutputVector& inputs_values) {
    if (get_output_partial_shape(0).is_dynamic() || is_const_fold_disabled()) {
        return false;
    }

    const auto& shape = get_output_shape(0);

    if (auto data_const = std::dynamic_pointer_cast<op::v0::Constant>(inputs_values[0].get_node_shared_ptr())) {
        output_values[0] = std::make_shared<op::v0::Constant>(*data_const, shape);
        return true;
    }
    return false;
}
