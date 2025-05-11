// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/one_hot.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "one_hot_shape_inference.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "openvino/reference/one_hot.hpp"

namespace ov {
namespace op {
namespace one_hot {
struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& indices,
                             const Shape& indices_shape,
                             char* const output_data,
                             const size_t output_et_size,
                             const int64_t one_hot_axis,
                             const char* const on_value,
                             const char* const off_value,
                             const int64_t axis) {
        reference::one_hot(indices.data<const T>(),
                           indices_shape,
                           output_data,
                           output_et_size,
                           one_hot_axis,
                           axis,
                           on_value,
                           off_value);
        return true;
    }
};
}  // namespace one_hot

namespace v1 {
OneHot::OneHot(const Output<Node>& indices,
               const Output<Node>& depth,
               const Output<Node>& on_value,
               const Output<Node>& off_value,
               int64_t axis)
    : Op({indices, depth, on_value, off_value}),
      m_axis(axis) {
    mark_as_precision_sensitive(input(1));
    constructor_validate_and_infer_types();
}

void OneHot::validate_and_infer_types() {
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

    std::vector<PartialShape> input_shapes = {indices_shape, depth_shape, on_value_shape, off_value_shape};
    resolve_axis(this);
    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, on_value_et, output_shapes[0]);
}

bool OneHot::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_OneHot_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    return true;
}

std::shared_ptr<Node> OneHot::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_OneHot_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<v1::OneHot>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), m_axis);
}

bool OneHot::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_OneHot_evaluate);
    OPENVINO_ASSERT(inputs.size() == 4 && outputs.size() == 1);

    const auto output_shape =
        shape_infer(this, ov::util::get_tensors_partial_shapes(inputs), make_tensor_accessor(inputs))
            .front()
            .to_shape();
    const auto axis = get_axis();
    OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < output_shape.size(), "Invalid axis value.");

    const auto depth = v0::Constant{inputs[1]}.cast_vector<int64_t>()[0];
    OPENVINO_ASSERT(static_cast<int64_t>(output_shape[axis]) == depth, "Incompatible axis and depth values.");

    const auto& indices = inputs[0];
    const auto& indices_shape = indices.get_shape();
    OPENVINO_ASSERT(shape_size(indices_shape) * depth == shape_size(output_shape),
                    "Incompatible I/O shapes or wrong depth value.");

    const auto on_value = static_cast<const char*>(inputs[2].data());
    const auto off_value = static_cast<const char*>(inputs[3].data());
    auto& output = outputs[0];
    output.set_shape(output_shape);
    using namespace ov::element;
    return IF_TYPE_OF(v1_OneHot_evaluate,
                      OV_PP_ET_LIST(i32, i64),
                      one_hot::Evaluate,
                      indices.get_element_type(),
                      indices,
                      indices_shape,
                      static_cast<char*>(output.data()),
                      output.get_element_type().size(),
                      output.get_shape()[axis],
                      on_value,
                      off_value,
                      axis);
}

bool OneHot::has_evaluate() const {
    OV_OP_SCOPE(v1_OneHot_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::i32:
    case element::i64:
        return true;
    default:
        return false;
    }
}

void OneHot::set_axis(int64_t axis) {
    m_axis = axis;
    resolve_axis(this);
}
}  // namespace v1
}  // namespace op
}  // namespace ov
