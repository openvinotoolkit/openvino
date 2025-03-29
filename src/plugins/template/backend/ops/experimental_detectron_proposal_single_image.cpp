// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/experimental_detectron_proposal_single_image.hpp"

#include "evaluate_node.hpp"
#include "evaluates_map.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    const auto attrs = op->get_attrs();

    size_t post_nms_count = 0;
    if (attrs.post_nms_count < 0) {
        OPENVINO_THROW("The attribute post_nms_count of the operation "
                       "ExperimentalDetectronGenerateProposalsSingleImage must be a "
                       "nonnegative integer.");
    } else {
        post_nms_count = static_cast<size_t>(attrs.post_nms_count);
    }

    const ov::Shape output_rois_shape = ov::Shape{post_nms_count, 4};
    const ov::Shape output_scores_shape = ov::Shape{post_nms_count};

    const auto output_type = op->get_input_element_type(0);

    const auto im_info_shape = inputs[0].get_shape();
    const auto anchors_shape = inputs[1].get_shape();
    const auto deltas_shape = inputs[2].get_shape();
    const auto scores_shape = inputs[3].get_shape();

    const auto im_info_data = get_floats(inputs[0], im_info_shape);
    const auto anchors_data = get_floats(inputs[1], anchors_shape);
    const auto deltas_data = get_floats(inputs[2], deltas_shape);
    const auto scores_data = get_floats(inputs[3], scores_shape);

    std::vector<float> output_rois(ov::shape_size(output_rois_shape));
    std::vector<float> output_scores(ov::shape_size(output_scores_shape));

    outputs[0].set_shape(output_rois_shape);
    outputs[1].set_shape(output_scores_shape);

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
    ov::reference::experimental_detectron_proposals_single_image_postprocessing(outputs[0].data(),
                                                                                outputs[1].data(),
                                                                                output_type,
                                                                                output_rois,
                                                                                output_scores,
                                                                                output_rois_shape,
                                                                                output_scores_shape);

    return true;
}

template <>
bool evaluate_node<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(std::shared_ptr<ov::Node> node,
                                                                                  ov::TensorVector& outputs,
                                                                                  const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(
            ov::as_type_ptr<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(
            ov::as_type_ptr<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(
            ov::as_type_ptr<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(
            ov::as_type_ptr<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(
            ov::as_type_ptr<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(
            ov::as_type_ptr<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(
            ov::as_type_ptr<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(
            ov::as_type_ptr<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(
            ov::as_type_ptr<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(
            ov::as_type_ptr<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(
            ov::as_type_ptr<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(
            ov::as_type_ptr<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(
            ov::as_type_ptr<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(
            ov::as_type_ptr<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(
            ov::as_type_ptr<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(
            ov::as_type_ptr<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(node),
            outputs,
            inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
