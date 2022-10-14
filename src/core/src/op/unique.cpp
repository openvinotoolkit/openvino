// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/unique.hpp"

#include "itt.hpp"
#include "ngraph/validation_util.hpp"
#include "unique_shape_inference.hpp"

namespace ov {
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
    // NODE_VALIDATION_CHECK(this,
    //                       get_input_element_type(1).is_real(),
    //                       "The element type of the grid input tensor must be a floating point type.");

    const std::vector<PartialShape> out_shapes(4);
    // shape_infer(this, {get_input_partial_shape(0), get_input_partial_shape(1)}, out_shapes);
    set_output_type(0, get_input_element_type(0), out_shapes[0]);
    set_output_type(0, get_input_element_type(0), out_shapes[1]);
    set_output_type(0, get_input_element_type(0), out_shapes[2]);
    set_output_type(0, get_input_element_type(0), out_shapes[3]);
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
