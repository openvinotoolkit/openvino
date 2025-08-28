// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/one_hot_base.hpp"

#include "itt.hpp"
#include "one_hot_shape_inference.hpp"

namespace ov {
namespace op {
namespace util {

OneHotBase::OneHotBase(const Output<Node>& indices,
                       const Output<Node>& depth,
                       const Output<Node>& on_value,
                       const Output<Node>& off_value,
                       int64_t axis)
    : Op({indices, depth, on_value, off_value}),
      m_axis(axis) {}

void OneHotBase::validate_and_infer_types() {
    OV_OP_SCOPE(util_OneHotBase_validate_and_infer_types);
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
    const auto output_shapes = shape_infer_base(this, input_shapes);

    set_output_type(0, on_value_et, output_shapes[0]);
}

bool OneHotBase::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(util_OneHotBase_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    return true;
}

void OneHotBase::set_axis(int64_t axis) {
    m_axis = axis;
    resolve_axis(this);
}

}  // namespace util
}  // namespace op
}  // namespace ov
