// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/unique.hpp"

#include "itt.hpp"
#include "ngraph/validation_util.hpp"
#include "openvino/op/util/op_types.hpp"
#include "unique_shape_inference.hpp"

namespace ov {
namespace {
int64_t extract_axis(const std::shared_ptr<op::v0::Constant>& axis_constant) {
    const auto axis_vec = axis_constant->cast_vector<int64_t>();
    return axis_vec.at(0);
}
}  // namespace

op::v10::Unique::Unique(const Output<Node>& data, const bool sorted, const element::Type& index_element_type)
    : op::Op{{data}},
      m_sorted{sorted},
      m_index_element_type{index_element_type} {
    constructor_validate_and_infer_types();
}

op::v10::Unique::Unique(const Output<Node>& data,
                        const Output<Node>& axis,
                        const bool sorted,
                        const element::Type& index_element_type)
    : op::Op{{data, axis}},
      m_sorted{sorted},
      m_index_element_type{index_element_type} {
    constructor_validate_and_infer_types();
}

bool op::v10::Unique::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v10_Unique_visit_attributes);
    visitor.on_attribute("sorted", m_sorted);
    visitor.on_attribute("index_element_type", m_index_element_type);
    return true;
}

void op::v10::Unique::validate_and_infer_types() {
    OV_OP_SCOPE(v10_Unique_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          m_index_element_type == element::i32 || m_index_element_type == element::i64,
                          "The element type of the outputs containing indices can only be set to i32 or i64");

    std::vector<PartialShape> out_shapes(4);
    if (get_input_size() == 1) {
        shape_infer(this, get_input_partial_shape(0), out_shapes);
    } else {
        NODE_VALIDATION_CHECK(
            this,
            get_input_element_type(1) == element::i32 || get_input_element_type(1) == element::i64,
            "The allowed element types of the 'axis' input tensor of the Unique operator are i32 and i64.");

        NODE_VALIDATION_CHECK(
            this,
            get_input_partial_shape(1) == Shape{} || get_input_partial_shape(1) == Shape{1},
            "The 'axis' input tensor of the Unique operator must be a scalar or 1D tensor with 1 element.");

        NODE_VALIDATION_CHECK(this,
                              ov::op::util::is_constant(input_value(1).get_node()),
                              "The 'axis' input of the Unique operator must be connected to a Constant.");
        const int64_t axis =
            extract_axis(std::dynamic_pointer_cast<op::v0::Constant>(input_value(1).get_node_shared_ptr()));

        shape_infer(this, get_input_partial_shape(0), out_shapes, std::unique_ptr<int64_t>{new int64_t{axis}});
    }

    set_output_type(0, get_input_element_type(0), out_shapes[0]);
    set_output_type(1, m_index_element_type, out_shapes[1]);
    set_output_type(2, m_index_element_type, out_shapes[2]);
    set_output_type(3, element::i64, out_shapes[3]);
}

std::shared_ptr<Node> op::v10::Unique::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v10_Unique_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 1) {
        return std::make_shared<op::v10::Unique>(new_args.at(0), this->get_sorted(), this->get_index_element_type());
    } else {
        return std::make_shared<op::v10::Unique>(new_args.at(0),
                                                 new_args.at(1),
                                                 this->get_sorted(),
                                                 this->get_index_element_type());
    }
}
}  // namespace ov
