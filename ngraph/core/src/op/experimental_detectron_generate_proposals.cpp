// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/experimental_detectron_generate_proposals.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v6::ExperimentalDetectronGenerateProposalsSingleImage,
                       "ExperimentalDetectronGenerateProposalsSingleImage",
                       6);

op::v6::ExperimentalDetectronGenerateProposalsSingleImage::
    ExperimentalDetectronGenerateProposalsSingleImage(const Output<Node>& im_info,
                                                      const Output<Node>& anchors,
                                                      const Output<Node>& deltas,
                                                      const Output<Node>& scores,
                                                      const Attributes& attrs)
    : Op({im_info, anchors, deltas, scores})
    , m_attrs(attrs)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v6::ExperimentalDetectronGenerateProposalsSingleImage::clone_with_new_inputs(
    const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v6_ExperimentalDetectronGenerateProposalsSingleImage_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(
        new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), m_attrs);
}

bool op::v6::ExperimentalDetectronGenerateProposalsSingleImage::visit_attributes(
    AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v6_ExperimentalDetectronGenerateProposalsSingleImage_visit_attributes);
    visitor.on_attribute("min_size", m_attrs.min_size);
    visitor.on_attribute("nms_threshold", m_attrs.nms_threshold);
    visitor.on_attribute("post_nms_count", m_attrs.post_nms_count);
    visitor.on_attribute("pre_nms_count", m_attrs.pre_nms_count);
    return true;
}

void op::v6::ExperimentalDetectronGenerateProposalsSingleImage::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v6_ExperimentalDetectronGenerateProposalsSingleImage_validate_and_infer_types);
    size_t post_nms_count = static_cast<size_t>(m_attrs.post_nms_count);
    auto input_et = get_input_element_type(0);

    set_output_size(2);
    set_output_type(0, input_et, Shape{post_nms_count, 4});
    set_output_type(1, input_et, Shape{post_nms_count});

    auto im_info_shape = get_input_partial_shape(0);
    auto anchors_shape = get_input_partial_shape(1);
    auto deltas_shape = get_input_partial_shape(2);
    auto scores_shape = get_input_partial_shape(3);

    if (im_info_shape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              im_info_shape.rank().get_length() == 1,
                              "The 'input_im_info' input is expected to be a 1D. Got: ",
                              im_info_shape);

        NODE_VALIDATION_CHECK(this,
                              im_info_shape[0].is_dynamic() || im_info_shape[0] == 3,
                              "The 'input_im_info' shape is expected to be a [3]. Got: ",
                              im_info_shape);
    }

    if (anchors_shape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              anchors_shape.rank().get_length() == 2,
                              "The 'input_anchors' input is expected to be a 2D. Got: ",
                              anchors_shape);

        NODE_VALIDATION_CHECK(this,
                              anchors_shape[1].is_dynamic() || anchors_shape[1] == 4,
                              "The second dimension of 'input_anchors' should be 4. Got: ",
                              anchors_shape[1]);
    }

    if (deltas_shape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              deltas_shape.rank().get_length() == 3,
                              "The 'input_deltas' input is expected to be a 3D. Got: ",
                              deltas_shape);
    }
    if (scores_shape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              scores_shape.rank().get_length() == 3,
                              "The 'input_scores' input is expected to be a 3D. Got: ",
                              scores_shape);
    }
    if (deltas_shape.rank().is_static() && scores_shape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              deltas_shape[1].is_dynamic() || scores_shape[1].is_dynamic() ||
                                  deltas_shape[1] == scores_shape[1],
                              "Heights for inputs 'input_deltas' and 'input_scores' should be "
                              "equal. Got: ",
                              deltas_shape[1],
                              scores_shape[1]);

        NODE_VALIDATION_CHECK(this,
                              deltas_shape[2].is_dynamic() || scores_shape[2].is_dynamic() ||
                                  deltas_shape[2] == scores_shape[2],
                              "Width for inputs 'input_deltas' and 'input_scores' should be "
                              "equal. Got: ",
                              deltas_shape[2],
                              scores_shape[2]);
    }
}
