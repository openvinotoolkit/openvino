//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/op/proposal.hpp"
#include "itt.hpp"

#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v0::Proposal, "Proposal", 0);

op::v0::Proposal::Proposal(const Output<Node>& class_probs,
                           const Output<Node>& bbox_deltas,
                           const Output<Node>& image_shape,
                           const ProposalAttrs& attrs)
    : Op({class_probs, bbox_deltas, image_shape})
    , m_attrs(attrs)
{
    constructor_validate_and_infer_types();
}

void op::v0::Proposal::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v0_Proposal_validate_and_infer_types);
    const auto& class_probs_pshape = get_input_partial_shape(0);
    const auto& class_bbox_deltas_pshape = get_input_partial_shape(1);
    const auto& image_shape_pshape = get_input_partial_shape(2);
    if (class_probs_pshape.is_static() && class_bbox_deltas_pshape.is_static() &&
        image_shape_pshape.is_static())
    {
        const Shape class_probs_shape{class_probs_pshape.to_shape()};
        const Shape class_bbox_deltas_shape{class_bbox_deltas_pshape.to_shape()};
        const Shape image_shape_shape{image_shape_pshape.to_shape()};

        NODE_VALIDATION_CHECK(
            this,
            class_probs_shape.size() == 4,
            "Proposal layer shape class_probs input must have rank 4 (class_probs_shape: ",
            class_probs_shape,
            ").");

        NODE_VALIDATION_CHECK(this,
                              class_bbox_deltas_shape.size() == 4,
                              "Proposal layer shape class_bbox_deltas_shape input must have rank 4 "
                              "(class_bbox_deltas_shape: ",
                              class_bbox_deltas_shape,
                              ").");

        NODE_VALIDATION_CHECK(
            this,
            image_shape_shape.size() == 1,
            "Proposal layer image_shape input must have rank 1 (image_shape_shape: ",
            image_shape_shape,
            ").");

        NODE_VALIDATION_CHECK(
            this,
            image_shape_shape[0] >= 3 && image_shape_shape[0] <= 4,
            "Image_shape 1D tensor must have => 3 and <= 4 elements (image_shape_shape[0]",
            image_shape_shape[0],
            ").");

        auto batch_size = class_probs_shape[0];
        set_output_type(0, get_input_element_type(0), Shape{batch_size * m_attrs.post_nms_topn, 5});
    }
    else
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
    }
}

shared_ptr<Node> op::v0::Proposal::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_Proposal_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v0::Proposal>(new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
}

bool op::v0::Proposal::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_Proposal_visit_attributes);
    visitor.on_attribute("base_size", m_attrs.base_size);
    visitor.on_attribute("pre_nms_topn", m_attrs.pre_nms_topn);
    visitor.on_attribute("post_nms_topn", m_attrs.post_nms_topn);
    visitor.on_attribute("nms_thresh", m_attrs.nms_thresh);
    visitor.on_attribute("feat_stride", m_attrs.feat_stride);
    visitor.on_attribute("min_size", m_attrs.min_size);
    visitor.on_attribute("ratio", m_attrs.ratio);
    visitor.on_attribute("scale", m_attrs.scale);
    visitor.on_attribute("clip_before_nms", m_attrs.clip_before_nms);
    visitor.on_attribute("clip_after_nms", m_attrs.clip_after_nms);
    visitor.on_attribute("normalize", m_attrs.normalize);
    visitor.on_attribute("box_size_scale", m_attrs.box_size_scale);
    visitor.on_attribute("box_coordinate_scale", m_attrs.box_coordinate_scale);
    visitor.on_attribute("framework", m_attrs.framework);
    return true;
}

NGRAPH_RTTI_DEFINITION(op::v4::Proposal, "Proposal", 4);

op::v4::Proposal::Proposal(const Output<Node>& class_probs,
                           const Output<Node>& class_bbox_deltas,
                           const Output<Node>& image_shape,
                           const op::ProposalAttrs& attrs)
    : v0::Proposal(class_probs, class_bbox_deltas, image_shape, attrs)
{
    constructor_validate_and_infer_types();
}

void op::v4::Proposal::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v4_Proposal_validate_and_infer_types);
    v0::Proposal::validate_and_infer_types();

    const auto& class_probs_pshape = get_input_partial_shape(0);
    const auto& class_bbox_deltas_pshape = get_input_partial_shape(1);
    const auto& image_shape_pshape = get_input_partial_shape(2);
    auto batch_size = class_probs_pshape[0];
    if (class_probs_pshape.is_static() && class_bbox_deltas_pshape.is_static() &&
        image_shape_pshape.is_static())
        set_output_type(
            1, get_input_element_type(0), PartialShape{batch_size * m_attrs.post_nms_topn});
    else
        set_output_type(1, get_input_element_type(0), PartialShape::dynamic());
}

std::shared_ptr<Node> op::v4::Proposal::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v4_Proposal_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v4::Proposal>(new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
}
