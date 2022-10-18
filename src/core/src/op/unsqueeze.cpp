// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/unsqueeze.hpp"

#include <cstddef>
#include <functional>
#include <set>

#include "itt.hpp"
#include "ngraph/runtime/reference/copy.hpp"
#include "ngraph/validation_util.hpp"
#include "unsqueeze_shape_inference.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v0::Unsqueeze);

namespace {
std::vector<PartialShape> get_node_input_partial_shapes(const ov::Node& node) {
    std::vector<PartialShape> out;
    out.reserve(node.get_input_size());
    for (size_t i = 0; i < node.get_input_size(); ++i) {
        out.emplace_back(node.get_input_partial_shape(i));
    }
    return out;
}
}  // namespace

op::v0::Unsqueeze::Unsqueeze(const Output<Node>& data, const Output<Node>& axes) : Op({data, axes}) {
    constructor_validate_and_infer_types();
}

void op::v0::Unsqueeze::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Unsqueeze_validate_and_infer_types);

    const auto& axes_pshape = get_input_partial_shape(AXES);

    NODE_VALIDATION_CHECK(this,
                          axes_pshape.rank().compatible(0) || axes_pshape.rank().compatible(1),
                          "Second input (axes) should not be of rank higher than 1. Got: ",
                          axes_pshape.rank().get_length());

    const auto input_shapes = get_node_input_partial_shapes(*this);
    auto output_shapes = std::vector<ov::PartialShape>(OUT_COUNT);

    shape_infer(this, input_shapes, output_shapes);

    set_output_size(output_shapes.size());
    set_output_type(OUT, get_input_element_type(ARG), output_shapes[OUT]);
}

bool op::v0::Unsqueeze::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Unsqueeze_visit_attributes);
    return true;
}

shared_ptr<Node> op::v0::Unsqueeze::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Unsqueeze_clone_with_new_inputs);
    if (new_args.size() != IN_COUNT) {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<Unsqueeze>(new_args.at(ARG), new_args.at(AXES));
}

namespace unsqueeze {
namespace {
template <element::Type_t ET>
bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out) {
    runtime::reference::copy(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), shape_size(out->get_shape()));
    return true;
}

bool evaluate_unsqueeze(const HostTensorPtr& arg0, const HostTensorPtr& arg1, const HostTensorPtr& out) {
    auto element_type = arg0->get_element_type();
    out->set_element_type(element_type);

    auto data_shape = arg0->get_shape();
    int64_t data_rank = static_cast<int64_t>(data_shape.size());
    auto axes_shape = arg1->get_shape();
    NGRAPH_CHECK(axes_shape.size() == 1 || axes_shape.empty(),
                 "Axes to add must be a scalar or 1D tensor with 1 element");

    auto out_shape = data_shape;
    int64_t out_rank = data_rank + static_cast<int64_t>(shape_size(axes_shape));
    // Get axes
    vector<int64_t> axes = read_index_vector(arg1);
    // Normalize axes
    std::transform(axes.begin(), axes.end(), axes.begin(), [out_rank](int64_t i) -> int64_t {
        return i < 0 ? out_rank + i : i;
    });
    // Sort in increasing order
    std::set<int64_t, less<int64_t>> axes_set(axes.begin(), axes.end());
    NGRAPH_CHECK(axes.size() == axes_set.size(), "Axes has duplicate axis.");
    for (int64_t axis : axes_set) {
        NGRAPH_CHECK(axis >= 0 && axis < out_rank, "Axis is out of bounds: ", axis);
        out_shape.insert(out_shape.begin() + axis, 1);
    }
    out->set_shape(out_shape);

    bool rc = true;
    switch (element_type) {
        NGRAPH_TYPE_CASE(evaluate_unsqueeze, i32, arg0, out);
        NGRAPH_TYPE_CASE(evaluate_unsqueeze, i64, arg0, out);
        NGRAPH_TYPE_CASE(evaluate_unsqueeze, u32, arg0, out);
        NGRAPH_TYPE_CASE(evaluate_unsqueeze, u64, arg0, out);
        NGRAPH_TYPE_CASE(evaluate_unsqueeze, f16, arg0, out);
        NGRAPH_TYPE_CASE(evaluate_unsqueeze, f32, arg0, out);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace unsqueeze

bool op::v0::Unsqueeze::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_Unsqueeze_evaluate);
    NGRAPH_CHECK(validate_host_tensor_vector(inputs, IN_COUNT));
    NGRAPH_CHECK(validate_host_tensor_vector(outputs, OUT_COUNT));
    return unsqueeze::evaluate_unsqueeze(inputs[ARG], inputs[AXES], outputs[OUT]);
}

bool op::v0::Unsqueeze::has_evaluate() const {
    OV_OP_SCOPE(v0_Unsqueeze_has_evaluate);
    switch (get_input_element_type(ARG)) {
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::f16:
    case ngraph::element::f32:
        return true;
    default:
        break;
    }
    return false;
}

bool op::v0::Unsqueeze::evaluate_lower(const HostTensorVector& output_values) const {
    if (!get_input_tensor(AXES).has_and_set_bound())
        return false;
    return default_lower_bound_evaluator(this, output_values);
}

bool op::v0::Unsqueeze::evaluate_upper(const HostTensorVector& output_values) const {
    if (!get_input_tensor(AXES).has_and_set_bound())
        return false;
    return default_upper_bound_evaluator(this, output_values);
}

bool op::v0::Unsqueeze::evaluate_label(TensorLabelVector& output_labels) const {
    if (!get_input_tensor(AXES).has_and_set_bound())
        return false;
    return default_label_evaluator(this, output_labels);
}

bool op::v0::Unsqueeze::constant_fold(OutputVector& output_values, const OutputVector& inputs_values) {
    if (get_output_partial_shape(ARG).is_dynamic() || is_const_fold_disabled()) {
        return false;
    }

    const auto& shape = get_output_shape(ARG);

    if (auto data_const = std::dynamic_pointer_cast<op::v0::Constant>(inputs_values[ARG].get_node_shared_ptr())) {
        output_values[OUT] = std::make_shared<op::v0::Constant>(*data_const, shape);
        return true;
    }
    return false;
}
