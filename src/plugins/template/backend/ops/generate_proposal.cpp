// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/generate_proposal.hpp"

#include "evaluate_node.hpp"
#include "evaluates_map.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v9::GenerateProposals>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    const auto& attrs = op->get_attrs();

    if (attrs.post_nms_count < 0) {
        OPENVINO_THROW("The attribute post_nms_count of the operation "
                       "GenerateProposals must be a "
                       "nonnegative integer.");
    }

    const auto& output_type = op->get_input_element_type(0);

    const auto& im_info_shape = inputs[0].get_shape();
    const auto& anchors_shape = inputs[1].get_shape();
    const auto& deltas_shape = inputs[2].get_shape();
    const auto& scores_shape = inputs[3].get_shape();

    const auto im_info_data = get_floats(inputs[0], im_info_shape);
    const auto anchors_data = get_floats(inputs[1], anchors_shape);
    const auto deltas_data = get_floats(inputs[2], deltas_shape);
    const auto scores_data = get_floats(inputs[3], scores_shape);

    std::vector<float> output_rois;
    std::vector<float> output_scores;
    std::vector<int64_t> output_num;

    ov::reference::generate_proposals(im_info_data,
                                      anchors_data,
                                      deltas_data,
                                      scores_data,
                                      attrs,
                                      im_info_shape,
                                      anchors_shape,
                                      deltas_shape,
                                      scores_shape,
                                      output_rois,
                                      output_scores,
                                      output_num);

    size_t num_selected = static_cast<size_t>(std::accumulate(output_num.begin(), output_num.end(), 0));

    ov::Shape output_rois_shape = ov::Shape{num_selected, 4};
    ov::Shape output_scores_shape = ov::Shape{num_selected};

    outputs[0].set_shape(output_rois_shape);
    outputs[1].set_shape(output_scores_shape);

    const auto& roi_num_type = op->get_output_element_type(2);
    ov::Shape output_roi_num_shape = ov::Shape{im_info_shape[0]};
    outputs[2].set_shape(output_roi_num_shape);

    ov::reference::generate_proposals_postprocessing(outputs[0].data(),
                                                     outputs[1].data(),
                                                     outputs[2].data(),
                                                     output_type,
                                                     roi_num_type,
                                                     output_rois,
                                                     output_scores,
                                                     output_num,
                                                     output_rois_shape,
                                                     output_scores_shape);

    return true;
}

template <>
bool evaluate_node<ov::op::v9::GenerateProposals>(std::shared_ptr<ov::Node> node,
                                                  ov::TensorVector& outputs,
                                                  const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v9::GenerateProposals>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v9::GenerateProposals>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v9::GenerateProposals>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v9::GenerateProposals>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v9::GenerateProposals>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v9::GenerateProposals>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v9::GenerateProposals>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v9::GenerateProposals>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v9::GenerateProposals>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v9::GenerateProposals>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v9::GenerateProposals>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v9::GenerateProposals>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v9::GenerateProposals>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v9::GenerateProposals>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v9::GenerateProposals>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v9::GenerateProposals>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
