// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "proposal_ie.hpp"

#include <memory>

#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::ProposalIE::type_info;

op::ProposalIE::ProposalIE(const Output<Node>& class_probs, const Output<Node>& class_logits,
                           const Output<Node>& image_shape, const ProposalAttrs& attrs)
    : Op({class_probs, class_logits, image_shape}), m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

void op::ProposalIE::validate_and_infer_types() {
    set_input_is_relevant_to_shape(2);

    const auto& class_probs_pshape = get_input_partial_shape(0);
    const auto& class_logits_pshape = get_input_partial_shape(1);
    const auto& image_shape_pshape = get_input_partial_shape(2);
    if (class_probs_pshape.is_static() && class_logits_pshape.is_static() && image_shape_pshape.is_static()) {
        const Shape class_probs_shape {class_probs_pshape.to_shape()};
        const Shape class_logits_shape {class_logits_pshape.to_shape()};
        const Shape image_shape_shape {image_shape_pshape.to_shape()};

        NODE_VALIDATION_CHECK(
            this, class_probs_shape.size() == 4,
            "Proposal layer shape class_probs input must have rank 4 (class_probs_shape: ", class_probs_shape, ").");

        NODE_VALIDATION_CHECK(this, class_logits_shape.size() == 4,
                              "Proposal layer shape class_logits_shape input must have rank 4 (class_logits_shape: ",
                              class_logits_shape, ").");

        NODE_VALIDATION_CHECK(
            this, image_shape_shape.size() == 2,
            "Proposal layer image_shape input must have rank 2 (image_shape_shape: ", image_shape_shape, ").");

        NODE_VALIDATION_CHECK(this, image_shape_shape[1] >= 3 && image_shape_shape[1] <= 4,
                              "Image_shape 1D tensor must have => 3 and <= 4 elements (image_shape_shape[1]",
                              image_shape_shape[1], ").");

        auto batch_size = class_probs_shape[0];
        set_output_type(0, get_input_element_type(0), Shape {batch_size * m_attrs.post_nms_topn, 5});
    } else {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
    }
}

shared_ptr<Node> op::ProposalIE::copy_with_new_args(const NodeVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<ProposalIE>(new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
}
