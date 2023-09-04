// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/experimental_detectron_proposal_single_image.hpp"

#include "evaluate_node.hpp"
#include "evaluates_map.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    const auto attrs = op->get_attrs();

    size_t post_nms_count = 0;
    if (attrs.post_nms_count < 0) {
        OPENVINO_THROW("The attribute post_nms_count of the operation "
                       "ExperimentalDetectronGenerateProposalsSingleImage must be a "
                       "nonnegative integer.");
    } else {
        post_nms_count = static_cast<size_t>(attrs.post_nms_count);
    }

    const ngraph::Shape output_rois_shape = ngraph::Shape{post_nms_count, 4};
    const ngraph::Shape output_scores_shape = ngraph::Shape{post_nms_count};

    const auto output_type = op->get_input_element_type(0);

    const auto im_info_shape = inputs[0]->get_shape();
    const auto anchors_shape = inputs[1]->get_shape();
    const auto deltas_shape = inputs[2]->get_shape();
    const auto scores_shape = inputs[3]->get_shape();

    const auto im_info_data = get_floats(inputs[0], im_info_shape);
    const auto anchors_data = get_floats(inputs[1], anchors_shape);
    const auto deltas_data = get_floats(inputs[2], deltas_shape);
    const auto scores_data = get_floats(inputs[3], scores_shape);

    std::vector<float> output_rois(ngraph::shape_size(output_rois_shape));
    std::vector<float> output_scores(ngraph::shape_size(output_scores_shape));

    outputs[0]->set_element_type(output_type);
    outputs[0]->set_shape(output_rois_shape);
    outputs[1]->set_element_type(output_type);
    outputs[1]->set_shape(output_scores_shape);

    ov::reference::experimental_detectron_proposals_single_image(im_info_data.data(),
                                                                 anchors_data.data(),
                                                                 deltas_data.data(),
                                                                 scores_data.data(),
                                                                 attrs,
                                                                 im_info_shape,
                                                                 anchors_shape,
                                                                 deltas_shape,
                                                                 scores_shape,
                                                                 output_rois.data(),
                                                                 output_scores.data());
    ov::reference::experimental_detectron_proposals_single_image_postprocessing(outputs[0]->get_data_ptr(),
                                                                                outputs[1]->get_data_ptr(),
                                                                                output_type,
                                                                                output_rois,
                                                                                output_scores,
                                                                                output_rois_shape,
                                                                                output_scores_shape);

    return true;
}

template <>
bool evaluate_node<ngraph::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(
    std::shared_ptr<ngraph::Node> node,
    const ngraph::HostTensorVector& outputs,
    const ngraph::HostTensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ngraph::op::v1::Select>(node) || ov::is_type<ngraph::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ngraph::element::Type_t::boolean:
        return evaluate<ngraph::element::Type_t::boolean>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::bf16:
        return evaluate<ngraph::element::Type_t::bf16>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::f16:
        return evaluate<ngraph::element::Type_t::f16>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::f64:
        return evaluate<ngraph::element::Type_t::f64>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::f32:
        return evaluate<ngraph::element::Type_t::f32>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::i4:
        return evaluate<ngraph::element::Type_t::i4>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::i8:
        return evaluate<ngraph::element::Type_t::i8>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::i16:
        return evaluate<ngraph::element::Type_t::i16>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::i32:
        return evaluate<ngraph::element::Type_t::i32>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::i64:
        return evaluate<ngraph::element::Type_t::i64>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::u1:
        return evaluate<ngraph::element::Type_t::u1>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::u4:
        return evaluate<ngraph::element::Type_t::u4>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::u8:
        return evaluate<ngraph::element::Type_t::u8>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::u16:
        return evaluate<ngraph::element::Type_t::u16>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::u32:
        return evaluate<ngraph::element::Type_t::u32>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::u64:
        return evaluate<ngraph::element::Type_t::u64>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}
