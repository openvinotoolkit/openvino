// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/one_hot.hpp"

#include <one_hot_shape_inference.hpp>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/runtime/reference/one_hot.hpp"
#include "ngraph/validation_util.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v1::OneHot);

op::v1::OneHot::OneHot(const Output<Node>& indices,
                       const Output<Node>& depth,
                       const Output<Node>& on_value,
                       const Output<Node>& off_value,
                       int64_t axis)
    : Op({indices, depth, on_value, off_value}),
      m_axis(axis) {
    ov::mark_as_precision_sensitive(input(1));
    constructor_validate_and_infer_types();
}

void op::v1::OneHot::validate_and_infer_types() {
    OV_OP_SCOPE(v1_OneHot_validate_and_infer_types);
    const auto& indices_et = get_input_element_type(0);
    const auto& depth_et = get_input_element_type(1);
    const auto& on_value_et = get_input_element_type(2);
    const auto& off_value_et = get_input_element_type(3);

    NODE_VALIDATION_CHECK(this,
                          indices_et.is_dynamic() || indices_et.is_integral(),
                          "Indices must be integral element type.");

    NODE_VALIDATION_CHECK(this,
                          depth_et.is_dynamic() || depth_et.is_integral(),
                          "Depth must be integral element type.");

    NODE_VALIDATION_CHECK(this,
                          on_value_et.compatible(off_value_et),
                          "on_value element type must be compatible with off_value element type.");

    const auto& indices_shape = get_input_partial_shape(0);
    const auto& depth_shape = get_input_partial_shape(1);
    const auto& on_value_shape = get_input_partial_shape(2);
    const auto& off_value_shape = get_input_partial_shape(3);

    std::vector<PartialShape> input_shapes = {indices_shape, depth_shape, on_value_shape, off_value_shape},
                              output_shapes = {PartialShape{}};
    resolve_axis(this);
    shape_infer(this, input_shapes, output_shapes);

    set_output_type(0, on_value_et, output_shapes[0]);
}

bool ngraph::op::v1::OneHot::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_OneHot_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    return true;
}

shared_ptr<Node> op::v1::OneHot::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_OneHot_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::OneHot>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), m_axis);
}

namespace one_hot {
namespace {
template <element::Type_t T>
bool evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values, const int64_t axis) {
    using INPUT_TYPE = typename element_type_traits<T>::value_type;
    const auto& indices = input_values[0];
    const auto& on_value = input_values[2];
    const auto& off_value = input_values[3];
    const auto& out = output_values[0];
    runtime::reference::one_hot<INPUT_TYPE>(indices->get_data_ptr<INPUT_TYPE>(),
                                            indices->get_shape(),
                                            out->get_data_ptr<char>(),
                                            out->get_element_type().size(),
                                            out->get_shape()[axis],
                                            axis,
                                            on_value->get_data_ptr<char>(),
                                            off_value->get_data_ptr<char>());
    return true;
}
bool evaluate_onehot(const HostTensorVector& output_values, const HostTensorVector& input_values, const int64_t axis) {
    bool rc = true;
    const auto& indices = input_values[0];
    switch (indices->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_onehot, i32, output_values, input_values, axis);
        NGRAPH_TYPE_CASE(evaluate_onehot, i64, output_values, input_values, axis);
    default:
        rc = false;
    }
    return rc;
}
}  // namespace
}  // namespace one_hot

bool op::v1::OneHot::evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const {
    OV_OP_SCOPE(v1_OneHot_evaluate);
    NGRAPH_CHECK(validate_host_tensor_vector(input_values, 4));
    NGRAPH_CHECK(validate_host_tensor_vector(output_values, 1));

    const auto& ind_Pshape = input_values[0]->get_partial_shape();
    const auto& out_Pshape = output_values[0]->get_partial_shape();
    NGRAPH_CHECK(ind_Pshape.is_static() && out_Pshape.is_static(), "Only static input/output shapes are supported");
    const auto out_shape = out_Pshape.get_shape();
    const int64_t axis = get_axis();
    NGRAPH_CHECK(axis >= 0 && static_cast<size_t>(axis) < out_shape.size(), "Invalid axis value.");
    const auto depth = std::make_shared<op::v0::Constant>(input_values[1])->cast_vector<int64_t>()[0];
    const auto ind_shape = ind_Pshape.get_shape();
    NGRAPH_CHECK(shape_size(ind_shape) * depth == shape_size(out_shape),
                 "Incompatible I/O shapes or wrong depth value.");
    NGRAPH_CHECK(static_cast<int64_t>(out_shape[axis]) == depth, "Incompatible axis and depth values.");
    return one_hot::evaluate_onehot(output_values, input_values, axis);
}

bool op::v1::OneHot::has_evaluate() const {
    OV_OP_SCOPE(v1_OneHot_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::i32:
    case ngraph::element::i64:
        return true;
    default:
        break;
    }
    return false;
}

void op::v1::OneHot::set_axis(int64_t axis) {
    m_axis = axis;
    resolve_axis(this);
}
