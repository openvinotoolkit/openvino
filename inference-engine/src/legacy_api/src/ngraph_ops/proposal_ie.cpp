// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/ngraph_ops/proposal_ie.hpp"

#include <memory>

#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::ProposalIE::type_info;

op::ProposalIE::ProposalIE(const Output<Node>& class_probs, const Output<Node>& class_bbox_deltas,
                           const Output<Node>& image_shape, const ProposalAttrs& attrs)
    : Op({class_probs, class_bbox_deltas, image_shape}), m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

void op::ProposalIE::validate_and_infer_types() {
    const auto& class_probs_pshape = get_input_partial_shape(0);
    const auto& class_bbox_deltas_pshape = get_input_partial_shape(1);
    const auto& image_shape_pshape = get_input_partial_shape(2);

    if (class_probs_pshape.is_static() && class_bbox_deltas_pshape.is_static() && image_shape_pshape.is_static()) {
        const Shape class_probs_shape {class_probs_pshape.to_shape()};
        const Shape class_bbox_deltas_shape {class_bbox_deltas_pshape.to_shape()};
        const Shape image_shape_shape {image_shape_pshape.to_shape()};

        NODE_VALIDATION_CHECK(
            this, class_probs_shape.size() == 4,
            "Proposal layer shape class_probs input must have rank 4 (class_probs_shape: ", class_probs_shape, ").");

        NODE_VALIDATION_CHECK(this, class_bbox_deltas_shape.size() == 4,
                              "Proposal layer shape class_bbox_deltas_shape input must have rank 4 (class_bbox_deltas_shape: ",
                              class_bbox_deltas_shape, ").");

        NODE_VALIDATION_CHECK(
            this, image_shape_shape.size() == 2,
            "Proposal layer image_shape input must have rank 2 (image_shape_shape: ", image_shape_shape, ").");

        NODE_VALIDATION_CHECK(this, image_shape_shape[1] >= 3 && image_shape_shape[1] <= 4,
                              "Image_shape 1D tensor must have => 3 and <= 4 elements (image_shape_shape[1]",
                              image_shape_shape[1], ").");

        auto batch_size = class_probs_shape[0];
        set_output_type(0, get_input_element_type(0), Shape {batch_size * m_attrs.post_nms_topn, 5});
        if (m_attrs.infer_probs)
            set_output_type(1, get_input_element_type(0), Shape {batch_size * m_attrs.post_nms_topn});
    } else {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
        if (m_attrs.infer_probs)
            set_output_type(1, get_input_element_type(0), PartialShape::dynamic());
    }
}

shared_ptr<Node> op::ProposalIE::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<ProposalIE>(new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
}

bool op::ProposalIE::visit_attributes(AttributeVisitor& visitor){
    visitor.on_attribute("ratio", m_attrs.ratio);
    visitor.on_attribute("scale", m_attrs.scale);
    visitor.on_attribute("base_size", m_attrs.base_size);
    visitor.on_attribute("pre_nms_topn", m_attrs.pre_nms_topn);
    visitor.on_attribute("post_nms_topn", m_attrs.post_nms_topn);
    visitor.on_attribute("nms_thresh", m_attrs.nms_thresh);
    visitor.on_attribute("feat_stride", m_attrs.feat_stride);
    visitor.on_attribute("min_size", m_attrs.min_size);
    visitor.on_attribute("box_size_scale", m_attrs.box_size_scale);
    visitor.on_attribute("box_coordinate_scale", m_attrs.box_coordinate_scale);
    visitor.on_attribute("clip_before_nms", m_attrs.clip_before_nms);
    visitor.on_attribute("clip_after_nms", m_attrs.clip_after_nms);
    visitor.on_attribute("normalize", m_attrs.normalize);
    visitor.on_attribute("framework", m_attrs.framework);
    return true;
}
