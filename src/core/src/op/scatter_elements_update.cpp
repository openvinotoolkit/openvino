// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scatter_elements_update.hpp"

#include <scatter_elements_update_shape_inference.hpp>

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"

using namespace std;

namespace ov {
op::v3::ScatterElementsUpdate::ScatterElementsUpdate(const Output<Node>& data,
                                                     const Output<Node>& indices,
                                                     const Output<Node>& updates,
                                                     const Output<Node>& axis)
    : ov::op::util::ScatterElementsUpdateBase(data, indices, updates, axis) {
    constructor_validate_and_infer_types();
}

bool op::v3::ScatterElementsUpdate::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v3_ScatterElementsUpdate_visit_attributes);
    return true;
}

shared_ptr<Node> op::v3::ScatterElementsUpdate::clone_with_new_inputs(const OutputVector& inputs) const {
    OV_OP_SCOPE(v3_ScatterElementsUpdate_clone_with_new_inputs);
    NODE_VALIDATION_CHECK(this,
                          inputs.size() == get_input_size(),
                          "clone_with_new_inputs() required inputs size: ",
                          get_input_size(),
                          "Got: ",
                          inputs.size());

    return make_shared<v3::ScatterElementsUpdate>(inputs.at(0), inputs.at(1), inputs.at(2), inputs.at(3));
}

op::v12::ScatterElementsUpdate::ScatterElementsUpdate(const Output<Node>& data,
                                                      const Output<Node>& indices,
                                                      const Output<Node>& updates,
                                                      const Output<Node>& axis,
                                                      const Reduction reduction,
                                                      const bool use_init_val)
    : op::util::ScatterElementsUpdateBase(data, indices, updates, axis),
      m_reduction{reduction},
      m_use_init_val{use_init_val} {
    constructor_validate_and_infer_types();
}

bool op::v12::ScatterElementsUpdate::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v12_ScatterElementsUpdate_visit_attributes);
    visitor.on_attribute("reduction", m_reduction);
    visitor.on_attribute("use_init_val", m_use_init_val);
    return true;
}

void op::v12::ScatterElementsUpdate::validate_and_infer_types() {
    OV_OP_SCOPE(v12_ScatterElementsUpdate_validate_and_infer_types);

    if (m_reduction == Reduction::MEAN) {
        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(0) != element::boolean,
                              "The 'mean' reduction type is not supported for boolean tensors");
    }

    ScatterElementsUpdateBase::validate_and_infer_types();
}

shared_ptr<Node> op::v12::ScatterElementsUpdate::clone_with_new_inputs(const OutputVector& inputs) const {
    OV_OP_SCOPE(v12_ScatterElementsUpdate_clone_with_new_inputs);
    NODE_VALIDATION_CHECK(this,
                          inputs.size() == get_input_size(),
                          "clone_with_new_inputs() required inputs size: ",
                          get_input_size(),
                          "Got: ",
                          inputs.size());

    return make_shared<v12::ScatterElementsUpdate>(inputs.at(0),
                                                   inputs.at(1),
                                                   inputs.at(2),
                                                   inputs.at(3),
                                                   m_reduction,
                                                   m_use_init_val);
}

bool op::v12::ScatterElementsUpdate::has_evaluate() const {
    if (m_reduction != Reduction::NONE) {
        return false;
    } else {
        return ScatterElementsUpdateBase::has_evaluate();
    }
}

template <>
OPENVINO_API EnumNames<op::v12::ScatterElementsUpdate::Reduction>&
EnumNames<op::v12::ScatterElementsUpdate::Reduction>::get() {
    static auto enum_names = EnumNames<op::v12::ScatterElementsUpdate::Reduction>(
        "op::v12::ScatterElementsUpdate::Reduction",
        {{"none", op::v12::ScatterElementsUpdate::Reduction::NONE},
         {"sum", op::v12::ScatterElementsUpdate::Reduction::SUM},
         {"prod", op::v12::ScatterElementsUpdate::Reduction::PROD},
         {"min", op::v12::ScatterElementsUpdate::Reduction::MIN},
         {"max", op::v12::ScatterElementsUpdate::Reduction::MAX},
         {"mean", op::v12::ScatterElementsUpdate::Reduction::MEAN}});
    return enum_names;
}
namespace op {
std::ostream& operator<<(std::ostream& s, const v12::ScatterElementsUpdate::Reduction& reduction) {
    return s << as_string(reduction);
}
}  // namespace op
}  // namespace ov
