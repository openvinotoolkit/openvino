// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/shape_of.hpp"

#include <algorithm>
#include <dimension_tracker.hpp>
#include <ngraph/validation_util.hpp>
#include <vector>

#include "itt.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/gather.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/shape_of.hpp"
#include "ngraph/type/element_type_traits.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v3::ShapeOf);

op::v3::ShapeOf::ShapeOf(const Output<Node>& arg, element::Type output_type) : Op({arg}), m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

void op::v3::ShapeOf::validate_and_infer_types() {
    OV_OP_SCOPE(v3_ShapeOf_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          m_output_type == element::i64 || m_output_type == element::i32,
                          "Output type must be i32 or i64");
    set_input_is_relevant_to_value(0, false);
    const auto input_partial_shape = get_input_partial_shape(0);
    set_output_type(0, m_output_type, ov::PartialShape{input_partial_shape.rank()});
}

bool ngraph::op::v3::ShapeOf::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v3_ShapeOf_visit_attributes);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

shared_ptr<Node> op::v3::ShapeOf::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v3_ShapeOf_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    auto new_shape_of = make_shared<op::v3::ShapeOf>(new_args.at(0), m_output_type);
    return new_shape_of;
}

namespace shape_of {
namespace {
template <element::Type_t ET>
inline bool evaluate(const ov::Shape& shape, const HostTensorPtr& output_value) {
    runtime::reference::shape_of(shape, output_value->get_data_ptr<ET>());
    return true;
}

bool evaluate_shape_of(const HostTensorPtr& output_value, const HostTensorPtr& input_value) {
    bool rc = true;
    ov::Shape shape = input_value->get_shape();
    output_value->set_shape(ov::Shape{shape.size()});
    switch (output_value->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_shape_of, i32, shape, output_value);
        NGRAPH_TYPE_CASE(evaluate_shape_of, i64, shape, output_value);
        NGRAPH_TYPE_CASE(evaluate_shape_of, u32, shape, output_value);
        NGRAPH_TYPE_CASE(evaluate_shape_of, u64, shape, output_value);
    default:
        rc = false;
        break;
    }
    return rc;
}

bool constant_fold_shape_of(Node* shape_of_node, Output<Node>& replacement, const Output<Node>& shape_of_input) {
    auto partial_shape = shape_of_input.get_partial_shape();
    auto output_type = shape_of_node->get_output_element_type(0);
    if (partial_shape.is_static()) {
        auto arg_shape = shape_of_input.get_shape();
        auto result_tensor = make_shared<HostTensor>(output_type, shape_of_node->get_output_shape(0));
        if (evaluate_shape_of(result_tensor, make_shared<HostTensor>(output_type, partial_shape))) {
            replacement = make_shared<op::Constant>(result_tensor);
            return true;
        }
        return false;
    }
    return false;
}

bool evaluate_bound_shape(const Node* shape_of_node, const HostTensorVector& output_values, bool is_upper) {
    NGRAPH_CHECK(shape_of_node, validate_host_tensor_vector(output_values, 1));
    const auto& input_partial_shape = shape_of_node->get_input_partial_shape(0);
    if (input_partial_shape.rank().is_dynamic())
        return false;
    const auto rank = input_partial_shape.rank().get_length();
    auto pshape_low = ov::PartialShape::dynamic(rank), pshape_up = ov::PartialShape::dynamic(rank);
    for (Dimension::value_type i = 0; i < rank; ++i) {
        Interval interval = input_partial_shape[i].get_interval();
        pshape_low[i] = interval.get_min_val();
        pshape_up[i] = Dimension(interval.get_max_val()).is_dynamic() ? Dimension(interval.get_max_val() - 1)
                                                                      : interval.get_max_val();
    }
    NGRAPH_CHECK(pshape_up.is_static() && pshape_low.is_static());
    const auto input_et = shape_of_node->get_input_element_type(0);
    const auto output_et = shape_of_node->get_output_element_type(0);
    if (pshape_low.to_shape() == pshape_up.to_shape()) {
        OPENVINO_SUPPRESS_DEPRECATED_START
        shape_of_node->evaluate(output_values, {std::make_shared<HostTensor>(input_et, pshape_low)});
        OPENVINO_SUPPRESS_DEPRECATED_END
        shape_of_node->get_output_tensor(0).set_lower_value(output_values[0]);
        shape_of_node->get_output_tensor(0).set_upper_value(output_values[0]);
    } else {
        HostTensorVector upper =
            is_upper ? output_values
                     : HostTensorVector{
                           std::make_shared<HostTensor>(output_et, ov::PartialShape{pshape_up.rank().get_length()})};
        OPENVINO_SUPPRESS_DEPRECATED_START
        shape_of_node->evaluate(upper, {std::make_shared<HostTensor>(input_et, pshape_up)});
        OPENVINO_SUPPRESS_DEPRECATED_END
        shape_of_node->get_output_tensor(0).set_upper_value(upper[0]);

        HostTensorVector lower =
            !is_upper ? output_values
                      : HostTensorVector{
                            std::make_shared<HostTensor>(output_et, ov::PartialShape{pshape_low.rank().get_length()})};
        OPENVINO_SUPPRESS_DEPRECATED_START
        shape_of_node->evaluate(lower, {std::make_shared<HostTensor>(input_et, pshape_low)});
        OPENVINO_SUPPRESS_DEPRECATED_END
        shape_of_node->get_output_tensor(0).set_lower_value(lower[0]);

        vector<bool> dynamic_mask;  // true if dimension is dynamic
        for (const auto& i : input_partial_shape)
            dynamic_mask.push_back(Dimension(i.get_interval().get_max_val()).is_dynamic());
        auto mask_const = ngraph::op::Constant::create(element::boolean, {dynamic_mask.size()}, dynamic_mask);
        auto dynamic_min_const = ngraph::op::Constant::create(output_et, {}, {0});
        auto dynamic_max_const = ngraph::op::Constant::create(
            output_et,
            {},
            {output_et == element::i64 ? std::numeric_limits<int64_t>::max() : std::numeric_limits<int32_t>::max()});

        op::v1::Select().evaluate(
            lower,
            {std::make_shared<HostTensor>(mask_const), std::make_shared<HostTensor>(dynamic_min_const), lower[0]});
        op::v1::Select().evaluate(
            upper,
            {std::make_shared<HostTensor>(mask_const), std::make_shared<HostTensor>(dynamic_max_const), upper[0]});
    }
    return true;
}

bool evaluate_label(const Node* shape_of_node, TensorLabelVector& output_labels) {
    const auto& shape = shape_of_node->get_input_partial_shape(0);
    NGRAPH_CHECK(shape.rank().is_static());  // sanity check. at this point value propagation was successful
    output_labels[0].reserve(shape.size());
    bool label_is_set = false;
    for (const auto& d : shape) {
        const auto& label = ov::DimensionTracker::get_label(d);
        if (label)
            label_is_set = true;
        output_labels[0].push_back(label);
    }
    return label_is_set;
}
}  // namespace
}  // namespace shape_of

bool op::v3::ShapeOf::evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const {
    OV_OP_SCOPE(v3_ShapeOf_evaluate);
    NGRAPH_CHECK(validate_host_tensor_vector(input_values, 1));
    NGRAPH_CHECK(validate_host_tensor_vector(output_values, 1));
    return shape_of::evaluate_shape_of(output_values[0], input_values[0]);
}

bool op::v3::ShapeOf::has_evaluate() const {
    OV_OP_SCOPE(v3_ShapeOf_has_evaluate);
    switch (get_output_element_type(0)) {
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64:
        return true;
    default:
        break;
    }
    return false;
}

bool op::v3::ShapeOf::evaluate_lower(const HostTensorVector& output_values) const {
    return shape_of::evaluate_bound_shape(this, output_values, false);
}

bool op::v3::ShapeOf::evaluate_upper(const HostTensorVector& output_values) const {
    return shape_of::evaluate_bound_shape(this, output_values, true);
}

bool op::v3::ShapeOf::evaluate_label(TensorLabelVector& output_labels) const {
    return shape_of::evaluate_label(this, output_labels);
}

bool op::v3::ShapeOf::constant_fold(OutputVector& output_values, const OutputVector& input_values) {
    OV_OP_SCOPE(v3_ShapeOf_constant_fold);
    if (is_const_fold_disabled()) {
        return false;
    }
    return shape_of::constant_fold_shape_of(this, output_values[0], input_values[0]);
}

// op::v0::ShapeOf
BWDCMP_RTTI_DEFINITION(op::v0::ShapeOf);

op::v0::ShapeOf::ShapeOf(const Output<Node>& arg) : Op({arg}) {
    constructor_validate_and_infer_types();
}

void op::v0::ShapeOf::validate_and_infer_types() {
    OV_OP_SCOPE(v0_ShapeOf_validate_and_infer_types);
    set_input_is_relevant_to_value(0, false);
    set_output_type(0, element::i64, ov::PartialShape{get_input_partial_shape(0).rank()});
}

bool ngraph::op::v0::ShapeOf::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_ShapeOf_visit_attributes);
    return true;
}

shared_ptr<Node> op::v0::ShapeOf::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_ShapeOf_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    auto new_shape_of = make_shared<op::v0::ShapeOf>(new_args.at(0));
    NGRAPH_CHECK(new_shape_of.get(),
                 new_shape_of != nullptr,
                 "Cannot clone ",
                 description(),
                 " operation with name ",
                 get_friendly_name());
    return new_shape_of;
}

bool op::v0::ShapeOf::evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const {
    OV_OP_SCOPE(v0_ShapeOf_evaluate);
    NGRAPH_CHECK(validate_host_tensor_vector(input_values, 1));
    NGRAPH_CHECK(validate_host_tensor_vector(output_values, 1));
    return shape_of::evaluate_shape_of(output_values[0], input_values[0]);
}

bool op::v0::ShapeOf::has_evaluate() const {
    OV_OP_SCOPE(v0_ShapeOf_has_evaluate);
    switch (get_output_element_type(0)) {
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64:
        return true;
    default:
        break;
    }
    return false;
}

bool op::v0::ShapeOf::constant_fold(OutputVector& output_values, const OutputVector& input_values) {
    OV_OP_SCOPE(v0_ShapeOf_constant_fold);
    if (is_const_fold_disabled()) {
        return false;
    }
    return shape_of::constant_fold_shape_of(this, output_values[0], input_values[0]);
}

bool op::v0::ShapeOf::evaluate_lower(const HostTensorVector& output_values) const {
    return shape_of::evaluate_bound_shape(this, output_values, false);
}

bool op::v0::ShapeOf::evaluate_upper(const HostTensorVector& output_values) const {
    return shape_of::evaluate_bound_shape(this, output_values, true);
}

bool op::v0::ShapeOf::evaluate_label(TensorLabelVector& output_labels) const {
    return shape_of::evaluate_label(this, output_labels);
}
