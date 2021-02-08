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
    const auto& class_probs_ps = get_input_partial_shape(0);
    const auto& bbox_deltas_ps = get_input_partial_shape(1);
    const auto& image_shape_ps = get_input_partial_shape(2);
    Dimension out_dim = Dimension::dynamic();
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).is_real(),
                          "Proposal layer input class_probs should have floating point type (",
                          get_input_element_type(0),
                          ").");

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(1).is_real(),
                          "Proposal layer input bbox_deltas should have floating point type (",
                          get_input_element_type(1),
                          ").");

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(2).is_real(),
                          "Proposal layer input image_shape should have floating point type (",
                          get_input_element_type(2),
                          ").");

    NODE_VALIDATION_CHECK(this,
                          class_probs_ps.rank().compatible(4),
                          "Proposal layer shape class_probs should be rank 4 compatible (",
                          class_probs_ps,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          bbox_deltas_ps.rank().compatible(4),
                          "Proposal layer shape bbox_deltas should be rank 4 compatible (",
                          bbox_deltas_ps,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          image_shape_ps.rank().compatible(1),
                          "Proposal layer shape image_shape should be rank 1 compatible (",
                          image_shape_ps,
                          ").");

    if (bbox_deltas_ps.is_static() && class_probs_ps.is_static())
    {
        // class probs and bbox deltas shapes are static, check anchor count and batch number
        // consistency
        NODE_VALIDATION_CHECK(this,
                              class_probs_ps[1].get_length() * 2 == bbox_deltas_ps[1].get_length(),
                              "Anchor number inconsistent between class_probs (",
                              class_probs_ps[1].get_length() / 2,
                              "), and bbox_deltas (",
                              bbox_deltas_ps[1].get_length() / 4,
                              ").");

        NODE_VALIDATION_CHECK(this,
                              class_probs_ps[0] == bbox_deltas_ps[0],
                              "Batch size inconsistent between class_probs (",
                              class_probs_ps[0],
                              ") and bbox deltas (",
                              bbox_deltas_ps[0],
                              ").");
    }

    if (image_shape_ps.is_static())
    {
        NODE_VALIDATION_CHECK(
            this,
            image_shape_ps[0].get_length() >= 3 && image_shape_ps[0].get_length() <= 4,
            "Image_shape 1D tensor must have => 3 and <= 4 elements (image_shape_shape[0]",
            image_shape_ps[0],
            ").");
    }

    if (class_probs_ps.rank().is_static() && bbox_deltas_ps.rank().is_static())
    {
        out_dim = (class_probs_ps[0] & bbox_deltas_ps[0]);
    }
    else if (class_probs_ps.rank().is_static())
    {
        out_dim = class_probs_ps[0];
    }
    else if (bbox_deltas_ps.rank().is_static())
    {
        out_dim = bbox_deltas_ps[0];
    }

    // intersect the batch size
    set_output_type(0, get_input_element_type(0), PartialShape{out_dim * m_attrs.post_nms_topn, 5});
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
    // Output shape was inferred in v0's validate_and_infer_types
    const auto proposals_ps = get_output_partial_shape(0);
    auto out_ps = PartialShape{Dimension::dynamic()};
    if (proposals_ps.rank().is_static() && proposals_ps.rank().compatible(2))
    {
        out_ps = PartialShape{proposals_ps[0]};
    }
    set_output_type(1, get_input_element_type(0), out_ps);
}

std::shared_ptr<Node> op::v4::Proposal::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v4_Proposal_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v4::Proposal>(new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
}
