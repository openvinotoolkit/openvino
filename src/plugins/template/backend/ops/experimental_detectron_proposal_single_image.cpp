// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"
#include "openvino/op/experimental_detectron_generate_proposals.hpp"

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
bool evaluate(const std::shared_ptr<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    const auto attrs = op->get_attrs();

    size_t post_nms_count = 0;
    if (attrs.post_nms_count < 0) {
        throw ngraph::ngraph_error("The attribute post_nms_count of the operation "
                                   "ExperimentalDetectronGenerateProposalsSingleImage must be a "
                                   "nonnegative integer.");
    } else {
        post_nms_count = static_cast<size_t>(attrs.post_nms_count);
    }

    const ov::Shape output_rois_shape = ov::Shape{post_nms_count, 4};
    const ov::Shape output_scores_shape = ov::Shape{post_nms_count};

    const auto output_type = op->get_input_element_type(0);

    const auto im_info_shape = inputs[0]->get_shape();
    const auto anchors_shape = inputs[1]->get_shape();
    const auto deltas_shape = inputs[2]->get_shape();
    const auto scores_shape = inputs[3]->get_shape();

    const auto im_info_data = get_floats(inputs[0], im_info_shape);
    const auto anchors_data = get_floats(inputs[1], anchors_shape);
    const auto deltas_data = get_floats(inputs[2], deltas_shape);
    const auto scores_data = get_floats(inputs[3], scores_shape);

    std::vector<float> output_rois(ov::shape_size(output_rois_shape));
    std::vector<float> output_scores(ov::shape_size(output_scores_shape));

    outputs[0]->set_element_type(output_type);
    outputs[0]->set_shape(output_rois_shape);
    outputs[1]->set_element_type(output_type);
    outputs[1]->set_shape(output_scores_shape);

    ngraph::runtime::reference::experimental_detectron_proposals_single_image(im_info_data.data(),
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
    ngraph::runtime::reference::experimental_detectron_proposals_single_image_postprocessing(outputs[0]->get_data_ptr(),
                                                                                             outputs[1]->get_data_ptr(),
                                                                                             output_type,
                                                                                             output_rois,
                                                                                             output_scores,
                                                                                             output_rois_shape,
                                                                                             output_scores_shape);

    return true;
}
