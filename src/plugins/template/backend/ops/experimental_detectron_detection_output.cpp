// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/experimental_detectron_detection_output.hpp"

#include "evaluate_node.hpp"
#include "evaluates_map.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v6::ExperimentalDetectronDetectionOutput>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    const auto attrs = op->get_attrs();
    size_t rois_num = attrs.max_detections_per_image;

    const ngraph::Shape output_boxes_shape = ngraph::Shape{rois_num, 4};
    const ngraph::Shape output_classes_shape = ngraph::Shape{rois_num};
    const ngraph::Shape output_scores_shape = ngraph::Shape{rois_num};

    const auto output_type = op->get_input_element_type(0);

    const auto boxes_data = get_floats(inputs[0], inputs[0]->get_shape());
    const auto input_deltas_data = get_floats(inputs[1], inputs[1]->get_shape());
    const auto input_scores_data = get_floats(inputs[2], inputs[2]->get_shape());
    const auto input_im_info_data = get_floats(inputs[3], inputs[3]->get_shape());

    std::vector<float> output_boxes(ngraph::shape_size(output_boxes_shape));
    std::vector<int32_t> output_classes(ngraph::shape_size(output_classes_shape));
    std::vector<float> output_scores(ngraph::shape_size(output_scores_shape));

    outputs[0]->set_element_type(output_type);
    outputs[0]->set_shape(output_boxes_shape);
    outputs[1]->set_element_type(ngraph::element::Type_t::i32);
    outputs[1]->set_shape(output_classes_shape);
    outputs[2]->set_element_type(output_type);
    outputs[2]->set_shape(output_scores_shape);

    ov::reference::experimental_detectron_detection_output(boxes_data.data(),
                                                           input_deltas_data.data(),
                                                           input_scores_data.data(),
                                                           input_im_info_data.data(),
                                                           attrs,
                                                           output_boxes.data(),
                                                           output_scores.data(),
                                                           output_classes.data());

    ov::reference::experimental_detectron_detection_output_postprocessing(outputs[0]->get_data_ptr(),
                                                                          outputs[1]->get_data_ptr(),
                                                                          outputs[2]->get_data_ptr(),
                                                                          output_type,
                                                                          output_boxes,
                                                                          output_classes,
                                                                          output_scores,
                                                                          output_boxes_shape,
                                                                          output_classes_shape,
                                                                          output_scores_shape);

    return true;
}

template <>
bool evaluate_node<ngraph::op::v6::ExperimentalDetectronDetectionOutput>(std::shared_ptr<ngraph::Node> node,
                                                                         const ngraph::HostTensorVector& outputs,
                                                                         const ngraph::HostTensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ngraph::op::v1::Select>(node) || ov::is_type<ngraph::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ngraph::element::Type_t::boolean:
        return evaluate<ngraph::element::Type_t::boolean>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronDetectionOutput>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::bf16:
        return evaluate<ngraph::element::Type_t::bf16>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronDetectionOutput>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::f16:
        return evaluate<ngraph::element::Type_t::f16>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronDetectionOutput>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::f64:
        return evaluate<ngraph::element::Type_t::f64>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronDetectionOutput>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::f32:
        return evaluate<ngraph::element::Type_t::f32>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronDetectionOutput>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::i4:
        return evaluate<ngraph::element::Type_t::i4>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronDetectionOutput>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::i8:
        return evaluate<ngraph::element::Type_t::i8>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronDetectionOutput>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::i16:
        return evaluate<ngraph::element::Type_t::i16>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronDetectionOutput>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::i32:
        return evaluate<ngraph::element::Type_t::i32>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronDetectionOutput>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::i64:
        return evaluate<ngraph::element::Type_t::i64>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronDetectionOutput>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::u1:
        return evaluate<ngraph::element::Type_t::u1>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronDetectionOutput>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::u4:
        return evaluate<ngraph::element::Type_t::u4>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronDetectionOutput>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::u8:
        return evaluate<ngraph::element::Type_t::u8>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronDetectionOutput>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::u16:
        return evaluate<ngraph::element::Type_t::u16>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronDetectionOutput>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::u32:
        return evaluate<ngraph::element::Type_t::u32>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronDetectionOutput>(node),
            outputs,
            inputs);
    case ngraph::element::Type_t::u64:
        return evaluate<ngraph::element::Type_t::u64>(
            ov::as_type_ptr<ngraph::op::v6::ExperimentalDetectronDetectionOutput>(node),
            outputs,
            inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}
