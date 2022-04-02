// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/multiclass_nms_ie_internal.hpp"

#include "../../core/shape_inference/include/multiclass_nms_shape_inference.hpp"
#include "itt.hpp"

using namespace std;
using namespace ngraph;
using namespace ov::op::util;

op::internal::MulticlassNmsIEInternal::MulticlassNmsIEInternal(const Output<Node>& boxes,
                                                               const Output<Node>& scores,
                                                               const ov::op::util::MulticlassNmsBase::Attributes& attrs)
    : opset9::MulticlassNms(boxes, scores, attrs) {
    constructor_validate_and_infer_types();
}

op::internal::MulticlassNmsIEInternal::MulticlassNmsIEInternal(const Output<Node>& boxes,
                                                               const Output<Node>& scores,
                                                               const Output<Node>& roisnum,
                                                               const ov::op::util::MulticlassNmsBase::Attributes& attrs)
    : opset9::MulticlassNms(boxes, scores, roisnum, attrs) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::internal::MulticlassNmsIEInternal::clone_with_new_inputs(
    const ngraph::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(internal_MulticlassNmsIEInternal_clone_with_new_inputs);

    if (new_args.size() == 3) {
        return std::make_shared<MulticlassNmsIEInternal>(new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
    } else if (new_args.size() == 2) {
        return std::make_shared<MulticlassNmsIEInternal>(new_args.at(0), new_args.at(1), m_attrs);
    }
    throw ngraph::ngraph_error("Unsupported number of inputs: " + std::to_string(new_args.size()));
}

void op::internal::MulticlassNmsIEInternal::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(internal_MulticlassNmsIEInternal_validate_and_infer_types);

    validate();

    const auto& boxes_ps = get_input_partial_shape(0);
    const auto& scores_ps = get_input_partial_shape(1);
    std::vector<PartialShape> input_shapes = {boxes_ps, scores_ps};
    if (get_input_size() == 3) {
        const auto& roisnum_ps = get_input_partial_shape(2);
        input_shapes.push_back(roisnum_ps);
    }

    std::vector<PartialShape> output_shapes = {{Dimension::dynamic(), 6},
                                               {Dimension::dynamic(), 1},
                                               {Dimension::dynamic()}};
    shape_infer(this, input_shapes, output_shapes, true, true);
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
    set_output_type(1, m_output_type, output_shapes[1]);
    set_output_type(2, m_output_type, output_shapes[2]);
}

const ::ngraph::Node::type_info_t& op::internal::MulticlassNmsIEInternal::get_type_info() const {
    return get_type_info_static();
}

const ::ngraph::Node::type_info_t& op::internal::MulticlassNmsIEInternal::get_type_info_static() {
    auto BaseNmsOpTypeInfoPtr = &opset9::MulticlassNms::get_type_info_static();

    static const std::string name = BaseNmsOpTypeInfoPtr->name;

    OPENVINO_SUPPRESS_DEPRECATED_START
    static const ::ngraph::Node::type_info_t type_info_static{name.c_str(),
                                                              BaseNmsOpTypeInfoPtr->version,
                                                              "ie_internal_opset",
                                                              BaseNmsOpTypeInfoPtr};
    OPENVINO_SUPPRESS_DEPRECATED_END
    return type_info_static;
}

#ifndef OPENVINO_STATIC_LIBRARY
const ::ngraph::Node::type_info_t op::internal::MulticlassNmsIEInternal::type_info =
    MulticlassNmsIEInternal::get_type_info_static();
#endif