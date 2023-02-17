// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"
#include "ngraph/runtime/reference/experimental_detectron_detection_output.hpp"
#include "openvino/op/experimental_detectron_detection_output.hpp"

std::vector<float> get_floats(const std::shared_ptr<ov::HostTensor>& input, const ov::Shape& shape) {
    size_t input_size = ov::shape_size(shape);
    std::vector<float> result(input_size);

    switch (input->get_element_type()) {
    case ov::element::Type_t::bf16: {
        ov::bfloat16* p = input->get_data_ptr<ov::bfloat16>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = float(p[i]);
        }
    } break;
    case ov::element::Type_t::f16: {
        ov::float16* p = input->get_data_ptr<ov::float16>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = float(p[i]);
        }
    } break;
    case ov::element::Type_t::f32: {
        float* p = input->get_data_ptr<float>();
        memcpy(result.data(), p, input_size * sizeof(float));
    } break;
    default:
        throw std::runtime_error("Unsupported data type.");
        break;
    }

    return result;
}

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v6::ExperimentalDetectronDetectionOutput>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    const auto attrs = op->get_attrs();
    size_t rois_num = attrs.max_detections_per_image;

    const ov::Shape output_boxes_shape = ov::Shape{rois_num, 4};
    const ov::Shape output_classes_shape = ov::Shape{rois_num};
    const ov::Shape output_scores_shape = ov::Shape{rois_num};

    const auto output_type = op->get_input_element_type(0);

    const auto boxes_data = get_floats(inputs[0], inputs[0]->get_shape());
    const auto input_deltas_data = get_floats(inputs[1], inputs[1]->get_shape());
    const auto input_scores_data = get_floats(inputs[2], inputs[2]->get_shape());
    const auto input_im_info_data = get_floats(inputs[3], inputs[3]->get_shape());

    std::vector<float> output_boxes(ov::shape_size(output_boxes_shape));
    std::vector<int32_t> output_classes(ov::shape_size(output_classes_shape));
    std::vector<float> output_scores(ov::shape_size(output_scores_shape));

    outputs[0]->set_element_type(output_type);
    outputs[0]->set_shape(output_boxes_shape);
    outputs[1]->set_element_type(ov::element::Type_t::i32);
    outputs[1]->set_shape(output_classes_shape);
    outputs[2]->set_element_type(output_type);
    outputs[2]->set_shape(output_scores_shape);

    ngraph::runtime::reference::experimental_detectron_detection_output(boxes_data.data(),
                                                                input_deltas_data.data(),
                                                                input_scores_data.data(),
                                                                input_im_info_data.data(),
                                                                attrs,
                                                                output_boxes.data(),
                                                                output_scores.data(),
                                                                output_classes.data());

    ngraph::runtime::reference::experimental_detectron_detection_output_postprocessing(outputs[0]->get_data_ptr(),
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
