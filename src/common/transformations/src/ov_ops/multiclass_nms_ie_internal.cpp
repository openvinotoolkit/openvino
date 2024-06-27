// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/multiclass_nms_ie_internal.hpp"

#include "../../core/shape_inference/include/multiclass_nms_shape_inference.hpp"
#include "itt.hpp"

using namespace std;
using namespace ov;

op::internal::MulticlassNmsIEInternal::MulticlassNmsIEInternal(const Output<Node>& boxes,
                                                               const Output<Node>& scores,
                                                               const op::util::MulticlassNmsBase::Attributes& attrs)
    : ov::op::v9::MulticlassNms(boxes, scores, attrs) {
    constructor_validate_and_infer_types();
}

op::internal::MulticlassNmsIEInternal::MulticlassNmsIEInternal(const Output<Node>& boxes,
                                                               const Output<Node>& scores,
                                                               const Output<Node>& roisnum,
                                                               const op::util::MulticlassNmsBase::Attributes& attrs)
    : ov::op::v9::MulticlassNms(boxes, scores, roisnum, attrs) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::internal::MulticlassNmsIEInternal::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(internal_MulticlassNmsIEInternal_clone_with_new_inputs);

    if (new_args.size() == 3) {
        return std::make_shared<MulticlassNmsIEInternal>(new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
    } else if (new_args.size() == 2) {
        return std::make_shared<MulticlassNmsIEInternal>(new_args.at(0), new_args.at(1), m_attrs);
    }
    OPENVINO_THROW("Unsupported number of inputs: " + std::to_string(new_args.size()));
}

void op::internal::MulticlassNmsIEInternal::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(internal_MulticlassNmsIEInternal_validate_and_infer_types);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);

    const auto output_shapes = shape_infer(this, input_shapes, false, true);

    validate();

    const auto& output_type = get_attrs().output_type;
    set_output_type(0, get_input_element_type(0), output_shapes[0].get_max_shape());
    set_output_type(1, output_type, output_shapes[1].get_max_shape());
    set_output_type(2, output_type, output_shapes[2]);
}
