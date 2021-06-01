// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/experimental_detectron_detection_output.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v6::ExperimentalDetectronDetectionOutput,
                       "ExperimentalDetectronDetectionOutput",
                       6);

op::v6::ExperimentalDetectronDetectionOutput::ExperimentalDetectronDetectionOutput(
    const Output<Node>& input_rois,
    const Output<Node>& input_deltas,
    const Output<Node>& input_scores,
    const Output<Node>& input_im_info,
    const Attributes& attrs)
    : Op({input_rois, input_deltas, input_scores, input_im_info})
    , m_attrs(attrs)
{
    constructor_validate_and_infer_types();
}

bool op::v6::ExperimentalDetectronDetectionOutput::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v6_ExperimentalDetectronDetectionOutput_visit_attributes);
    visitor.on_attribute("score_threshold", m_attrs.score_threshold);
    visitor.on_attribute("nms_threshold", m_attrs.nms_threshold);
    visitor.on_attribute("max_delta_log_wh", m_attrs.max_delta_log_wh);
    visitor.on_attribute("num_classes", m_attrs.num_classes);
    visitor.on_attribute("post_nms_count", m_attrs.post_nms_count);
    visitor.on_attribute("max_detections_per_image", m_attrs.max_detections_per_image);
    visitor.on_attribute("class_agnostic_box_regression", m_attrs.class_agnostic_box_regression);
    visitor.on_attribute("deltas_weights", m_attrs.deltas_weights);
    return true;
}

void op::v6::ExperimentalDetectronDetectionOutput::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v6_ExperimentalDetectronDetectionOutput_validate_and_infer_types);
    size_t rois_num = m_attrs.max_detections_per_image;
    auto input_et = get_input_element_type(0);

    auto rois_shape = get_input_partial_shape(0);
    auto deltas_shape = get_input_partial_shape(1);
    auto scores_shape = get_input_partial_shape(2);
    auto im_info_shape = get_input_partial_shape(3);

    set_output_size(3);
    set_output_type(0, input_et, Shape{rois_num, 4});
    set_output_type(1, element::Type_t::i32, Shape{rois_num});
    set_output_type(2, input_et, Shape{rois_num});

    NODE_VALIDATION_CHECK(
        this, rank_is_dynamic_or_equal_to(rois_shape, 2), "Input rois rank must be equal to 2.");

    NODE_VALIDATION_CHECK(this,
                          dims_are_equal(rois_shape, 1, 4),
                          "The last dimension of the 'input_rois' input must be equal to 4. Got: ",
                          rois_shape);

    NODE_VALIDATION_CHECK(
        this, rank_is_dynamic_or_equal_to(rois_shape, 2), "Input deltas rank must be equal to 2.");

    NODE_VALIDATION_CHECK(this,
                          dims_are_equal(deltas_shape, 1, m_attrs.num_classes * 4),
                          "The last dimension of the 'input_deltas' input must be equal to the "
                          "value of the attribute 'num_classes' * 4. Got: ",
                          deltas_shape);

    NODE_VALIDATION_CHECK(this,
                          rank_is_dynamic_or_equal_to(scores_shape, 2),
                          "Input scores rank must be equal to 2.");

    NODE_VALIDATION_CHECK(this,
                          dims_are_equal(scores_shape, 1, m_attrs.num_classes),
                          "The last dimension of the 'input_scores' input must be equal to the "
                          "value of the attribute 'num_classes'. Got: ",
                          scores_shape);

    NODE_VALIDATION_CHECK(this,
                          rank_is_dynamic_or_equal_to(im_info_shape, 2),
                          "Input image info rank must be equal to 2.");

    NODE_VALIDATION_CHECK(this,
                          dims_are_equal(rois_shape, 0, deltas_shape, 0) &&
                              dims_are_equal(deltas_shape, 0, scores_shape, 0),
                          "The first dimension of inputs 'input_rois', 'input_deltas', "
                          "'input_scores' must be the same. input_rois shape: ",
                          rois_shape,
                          "; input_deltas shape: ",
                          deltas_shape,
                          "; input_scores shape: ",
                          scores_shape);
}

shared_ptr<Node> op::v6::ExperimentalDetectronDetectionOutput::clone_with_new_inputs(
    const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v6_ExperimentalDetectronDetectionOutput_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v6::ExperimentalDetectronDetectionOutput>(
        new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), m_attrs);
}
