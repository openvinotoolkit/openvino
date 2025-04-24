// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/experimental_detectron_detection_output.hpp"

#include "evaluate_node.hpp"
#include "evaluates_map.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v6::ExperimentalDetectronDetectionOutput>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    const auto attrs = op->get_attrs();
    size_t rois_num = attrs.max_detections_per_image;

    const ov::Shape output_boxes_shape = ov::Shape{rois_num, 4};
    const ov::Shape output_classes_shape = ov::Shape{rois_num};
    const ov::Shape output_scores_shape = ov::Shape{rois_num};

    const auto output_type = op->get_input_element_type(0);

    const auto boxes_data = get_floats(inputs[0], inputs[0].get_shape());
    const auto input_deltas_data = get_floats(inputs[1], inputs[1].get_shape());
    const auto input_scores_data = get_floats(inputs[2], inputs[2].get_shape());
    const auto input_im_info_data = get_floats(inputs[3], inputs[3].get_shape());

    std::vector<float> output_boxes(ov::shape_size(output_boxes_shape));
    std::vector<int32_t> output_classes(ov::shape_size(output_classes_shape));
    std::vector<float> output_scores(ov::shape_size(output_scores_shape));

    outputs[0].set_shape(output_boxes_shape);
    outputs[1].set_shape(output_classes_shape);
    outputs[2].set_shape(output_scores_shape);

    ov::reference::experimental_detectron_detection_output(boxes_data.data(),
                                                           input_deltas_data.data(),
                                                           input_scores_data.data(),
                                                           input_im_info_data.data(),
                                                           attrs,
                                                           output_boxes.data(),
                                                           output_scores.data(),
                                                           output_classes.data());

    ov::reference::experimental_detectron_detection_output_postprocessing(outputs[0].data(),
                                                                          outputs[1].data(),
                                                                          outputs[2].data(),
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
bool evaluate_node<ov::op::v6::ExperimentalDetectronDetectionOutput>(std::shared_ptr<ov::Node> node,
                                                                     ov::TensorVector& outputs,
                                                                     const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronDetectionOutput>(node),
                                              outputs,
                                              inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronDetectionOutput>(node),
                                           outputs,
                                           inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronDetectionOutput>(node),
                                          outputs,
                                          inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronDetectionOutput>(node),
                                          outputs,
                                          inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronDetectionOutput>(node),
                                          outputs,
                                          inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronDetectionOutput>(node),
                                         outputs,
                                         inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronDetectionOutput>(node),
                                         outputs,
                                         inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronDetectionOutput>(node),
                                          outputs,
                                          inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronDetectionOutput>(node),
                                          outputs,
                                          inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronDetectionOutput>(node),
                                          outputs,
                                          inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronDetectionOutput>(node),
                                         outputs,
                                         inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronDetectionOutput>(node),
                                         outputs,
                                         inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronDetectionOutput>(node),
                                         outputs,
                                         inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronDetectionOutput>(node),
                                          outputs,
                                          inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronDetectionOutput>(node),
                                          outputs,
                                          inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v6::ExperimentalDetectronDetectionOutput>(node),
                                          outputs,
                                          inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
