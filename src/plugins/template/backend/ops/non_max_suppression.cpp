// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/non_max_suppression.hpp"

#include "evaluates_map.hpp"
#include "ngraph/runtime/reference/utils/nms_common.hpp"
#include "openvino/op/non_max_suppression.hpp"

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

std::vector<int64_t> get_integers(const std::shared_ptr<ov::HostTensor>& input, const ov::Shape& shape) {
    size_t input_size = ov::shape_size(shape);
    std::vector<int64_t> result(input_size);

    switch (input->get_element_type()) {
    case ov::element::Type_t::i8: {
        auto p = input->get_data_ptr<int8_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ov::element::Type_t::i16: {
        auto p = input->get_data_ptr<int16_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ov::element::Type_t::i32: {
        auto p = input->get_data_ptr<int32_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ov::element::Type_t::i64: {
        auto p = input->get_data_ptr<int64_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ov::element::Type_t::u8: {
        auto p = input->get_data_ptr<uint8_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ov::element::Type_t::u16: {
        auto p = input->get_data_ptr<uint16_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ov::element::Type_t::u32: {
        auto p = input->get_data_ptr<uint32_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ov::element::Type_t::u64: {
        auto p = input->get_data_ptr<uint64_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    default:
        throw std::runtime_error("Unsupported data type in op NonMaxSuppression-5");
        break;
    }

    return result;
}

namespace nms_v5 {
using V5BoxEncoding = ov::op::v5::NonMaxSuppression::BoxEncodingType;

struct InfoForNMS5 {
    int64_t max_output_boxes_per_class;
    float iou_threshold;
    float score_threshold;
    float soft_nms_sigma;
    ov::Shape out_shape;
    ov::Shape boxes_shape;
    ov::Shape scores_shape;
    std::vector<float> boxes_data;
    std::vector<float> scores_data;
    size_t out_shape_size;
    bool sort_result_descending;
    ov::element::Type output_type;
};

constexpr size_t boxes_port = 0;
constexpr size_t scores_port = 1;

ov::PartialShape infer_selected_indices_shape(const std::vector<std::shared_ptr<ov::HostTensor>>& inputs,
                                              int64_t max_output_boxes_per_class) {
    const auto boxes_ps = inputs[boxes_port]->get_partial_shape();
    const auto scores_ps = inputs[scores_port]->get_partial_shape();

    // NonMaxSuppression produces triplets
    // that have the following format: [batch_index, class_index, box_index]
    ov::PartialShape result = {ov::Dimension::dynamic(), 3};

    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static()) {
        const auto num_boxes_boxes = boxes_ps[1];
        if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static()) {
            const auto num_boxes = num_boxes_boxes.get_length();
            const auto num_classes = scores_ps[1].get_length();

            result[0] = std::min(num_boxes, max_output_boxes_per_class) * num_classes * scores_ps[0].get_length();
        }
    }
    return result;
}

void normalize_corner(float* boxes, const ov::Shape& boxes_shape) {
    size_t total_num_of_boxes = ov::shape_size(boxes_shape) / 4;
    for (size_t i = 0; i < total_num_of_boxes; ++i) {
        float* current_box = boxes + 4 * i;

        float y1 = current_box[0];
        float x1 = current_box[1];
        float y2 = current_box[2];
        float x2 = current_box[3];

        float ymin = std::min(y1, y2);
        float ymax = std::max(y1, y2);
        float xmin = std::min(x1, x2);
        float xmax = std::max(x1, x2);

        current_box[0] = ymin;
        current_box[1] = xmin;
        current_box[2] = ymax;
        current_box[3] = xmax;
    }
}

void normalize_center(float* boxes, const ov::Shape& boxes_shape) {
    size_t total_num_of_boxes = ov::shape_size(boxes_shape) / 4;
    for (size_t i = 0; i < total_num_of_boxes; ++i) {
        float* current_box = boxes + 4 * i;

        float x_center = current_box[0];
        float y_center = current_box[1];
        float width = current_box[2];
        float height = current_box[3];

        float y1 = y_center - height / 2.0f;
        float x1 = x_center - width / 2.0f;
        float y2 = y_center + height / 2.0f;
        float x2 = x_center + width / 2.0f;

        current_box[0] = y1;
        current_box[1] = x1;
        current_box[2] = y2;
        current_box[3] = x2;
    }
}

void normalize_box_encoding(float* boxes, const ov::Shape& boxes_shape, const V5BoxEncoding box_encoding) {
    if (box_encoding == V5BoxEncoding::CORNER) {
        normalize_corner(boxes, boxes_shape);
    } else {
        normalize_center(boxes, boxes_shape);
    }
}

std::vector<float> prepare_boxes_data(const std::shared_ptr<ov::HostTensor>& boxes,
                                      const ov::Shape& boxes_shape,
                                      const V5BoxEncoding box_encoding) {
    auto result = get_floats(boxes, boxes_shape);
    normalize_box_encoding(result.data(), boxes_shape, box_encoding);
    return result;
}

std::vector<float> prepare_scores_data(const std::shared_ptr<ov::HostTensor>& scores, const ov::Shape& scores_shape) {
    auto result = get_floats(scores, scores_shape);
    return result;
}

InfoForNMS5 get_info_for_nms5_eval(const std::shared_ptr<ov::op::v5::NonMaxSuppression>& nms5,
                                   const std::vector<std::shared_ptr<ov::HostTensor>>& inputs) {
    InfoForNMS5 result;

    result.max_output_boxes_per_class = inputs.size() > 2 ? get_integers(inputs[2], ov::Shape({}))[0] : 0;
    result.iou_threshold = inputs.size() > 3 ? get_floats(inputs[3], ov::Shape({}))[0] : 0.0f;
    result.score_threshold = inputs.size() > 4 ? get_floats(inputs[4], ov::Shape({}))[0] : 0.0f;
    result.soft_nms_sigma = inputs.size() > 5 ? get_floats(inputs[5], ov::Shape({}))[0] : 0.0f;

    auto selected_indices_shape = infer_selected_indices_shape(inputs, result.max_output_boxes_per_class);
    result.out_shape = selected_indices_shape.to_shape();

    result.boxes_shape = inputs[boxes_port]->get_shape();
    result.scores_shape = inputs[scores_port]->get_shape();

    result.boxes_data = prepare_boxes_data(inputs[boxes_port], result.boxes_shape, nms5->get_box_encoding());
    result.scores_data = prepare_scores_data(inputs[scores_port], result.scores_shape);

    result.out_shape_size = ov::shape_size(result.out_shape);

    result.sort_result_descending = nms5->get_sort_result_descending();

    result.output_type = nms5->get_output_type();

    return result;
}
}  // namespace nms_v5

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v5::NonMaxSuppression>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    auto info = nms_v5::get_info_for_nms5_eval(op, inputs);

    std::vector<int64_t> selected_indices(info.out_shape_size);
    std::vector<float> selected_scores(info.out_shape_size);
    int64_t valid_outputs = 0;

    ngraph::runtime::reference::non_max_suppression5(info.boxes_data.data(),
                                                     info.boxes_shape,
                                                     info.scores_data.data(),
                                                     info.scores_shape,
                                                     info.max_output_boxes_per_class,
                                                     info.iou_threshold,
                                                     info.score_threshold,
                                                     info.soft_nms_sigma,
                                                     selected_indices.data(),
                                                     info.out_shape,
                                                     selected_scores.data(),
                                                     info.out_shape,
                                                     &valid_outputs,
                                                     info.sort_result_descending);

    auto selected_scores_type = (outputs.size() < 3) ? ov::element::f32 : outputs[1]->get_element_type();

    ngraph::runtime::reference::nms_postprocessing(outputs,
                                                   info.output_type,
                                                   selected_indices,
                                                   selected_scores,
                                                   valid_outputs,
                                                   selected_scores_type);
    return true;
}

namespace nms_v9 {
using V9BoxEncoding = ov::op::v9::NonMaxSuppression::BoxEncodingType;

struct InfoForNMS9 {
    int64_t max_output_boxes_per_class;
    float iou_threshold;
    float score_threshold;
    float soft_nms_sigma;
    ov::Shape out_shape;
    ov::Shape boxes_shape;
    ov::Shape scores_shape;
    std::vector<float> boxes_data;
    std::vector<float> scores_data;
    size_t out_shape_size;
    bool sort_result_descending;
    ov::element::Type output_type;
};

constexpr size_t boxes_port = 0;
constexpr size_t scores_port = 1;

ov::PartialShape infer_selected_indices_shape(const std::vector<std::shared_ptr<ov::HostTensor>>& inputs,
                                              int64_t max_output_boxes_per_class) {
    const auto boxes_ps = inputs[boxes_port]->get_partial_shape();
    const auto scores_ps = inputs[scores_port]->get_partial_shape();

    // NonMaxSuppression produces triplets
    // that have the following format: [batch_index, class_index, box_index]
    ov::PartialShape result = {ov::Dimension::dynamic(), 3};

    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static()) {
        const auto num_boxes_boxes = boxes_ps[1];
        if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static()) {
            const auto num_boxes = num_boxes_boxes.get_length();
            const auto num_classes = scores_ps[1].get_length();

            result[0] = std::min(num_boxes, max_output_boxes_per_class) * num_classes * scores_ps[0].get_length();
        }
    }
    return result;
}

void normalize_corner(float* boxes, const ov::Shape& boxes_shape) {
    size_t total_num_of_boxes = ov::shape_size(boxes_shape) / 4;
    for (size_t i = 0; i < total_num_of_boxes; ++i) {
        float* current_box = boxes + 4 * i;

        float y1 = current_box[0];
        float x1 = current_box[1];
        float y2 = current_box[2];
        float x2 = current_box[3];

        float ymin = std::min(y1, y2);
        float ymax = std::max(y1, y2);
        float xmin = std::min(x1, x2);
        float xmax = std::max(x1, x2);

        current_box[0] = ymin;
        current_box[1] = xmin;
        current_box[2] = ymax;
        current_box[3] = xmax;
    }
}

void normalize_center(float* boxes, const ov::Shape& boxes_shape) {
    size_t total_num_of_boxes = ov::shape_size(boxes_shape) / 4;
    for (size_t i = 0; i < total_num_of_boxes; ++i) {
        float* current_box = boxes + 4 * i;

        float x_center = current_box[0];
        float y_center = current_box[1];
        float width = current_box[2];
        float height = current_box[3];

        float y1 = y_center - height / 2.0;
        float x1 = x_center - width / 2.0;
        float y2 = y_center + height / 2.0;
        float x2 = x_center + width / 2.0;

        current_box[0] = y1;
        current_box[1] = x1;
        current_box[2] = y2;
        current_box[3] = x2;
    }
}

void normalize_box_encoding(float* boxes, const ov::Shape& boxes_shape, const V9BoxEncoding box_encoding) {
    if (box_encoding == V9BoxEncoding::CORNER) {
        normalize_corner(boxes, boxes_shape);
    } else {
        normalize_center(boxes, boxes_shape);
    }
}

std::vector<float> prepare_boxes_data(const std::shared_ptr<ov::HostTensor>& boxes,
                                      const ov::Shape& boxes_shape,
                                      const V9BoxEncoding box_encoding) {
    auto result = get_floats(boxes, boxes_shape);
    normalize_box_encoding(result.data(), boxes_shape, box_encoding);
    return result;
}

std::vector<float> prepare_scores_data(const std::shared_ptr<ov::HostTensor>& scores, const ov::Shape& scores_shape) {
    auto result = get_floats(scores, scores_shape);
    return result;
}

InfoForNMS9 get_info_for_nms9_eval(const std::shared_ptr<ov::op::v9::NonMaxSuppression>& nms9,
                                   const std::vector<std::shared_ptr<ov::HostTensor>>& inputs) {
    InfoForNMS9 result;

    result.max_output_boxes_per_class = inputs.size() > 2 ? get_integers(inputs[2], ov::Shape({}))[0] : 0;
    result.iou_threshold = inputs.size() > 3 ? get_floats(inputs[3], ov::Shape({}))[0] : 0.0f;
    result.score_threshold = inputs.size() > 4 ? get_floats(inputs[4], ov::Shape({}))[0] : 0.0f;
    result.soft_nms_sigma = inputs.size() > 5 ? get_floats(inputs[5], ov::Shape({}))[0] : 0.0f;

    auto selected_indices_shape = infer_selected_indices_shape(inputs, result.max_output_boxes_per_class);
    result.out_shape = selected_indices_shape.to_shape();

    result.boxes_shape = inputs[boxes_port]->get_shape();
    result.scores_shape = inputs[scores_port]->get_shape();

    result.boxes_data = prepare_boxes_data(inputs[boxes_port], result.boxes_shape, nms9->get_box_encoding());
    result.scores_data = prepare_scores_data(inputs[scores_port], result.scores_shape);

    result.out_shape_size = shape_size(result.out_shape);

    result.sort_result_descending = nms9->get_sort_result_descending();

    result.output_type = nms9->get_output_type();

    return result;
}
}  // namespace nms_v9

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v9::NonMaxSuppression>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    auto info = nms_v9::get_info_for_nms9_eval(op, inputs);

    std::vector<int64_t> selected_indices(info.out_shape_size);
    std::vector<float> selected_scores(info.out_shape_size);
    int64_t valid_outputs = 0;

    ngraph::runtime::reference::non_max_suppression(info.boxes_data.data(),
                                                    info.boxes_shape,
                                                    info.scores_data.data(),
                                                    info.scores_shape,
                                                    info.max_output_boxes_per_class,
                                                    info.iou_threshold,
                                                    info.score_threshold,
                                                    info.soft_nms_sigma,
                                                    selected_indices.data(),
                                                    info.out_shape,
                                                    selected_scores.data(),
                                                    info.out_shape,
                                                    &valid_outputs,
                                                    info.sort_result_descending);

    auto selected_scores_type = (outputs.size() < 3) ? ov::element::f32 : outputs[1]->get_element_type();

    ngraph::runtime::reference::nms_postprocessing(outputs,
                                                   info.output_type,
                                                   selected_indices,
                                                   selected_scores,
                                                   valid_outputs,
                                                   selected_scores_type);
    return true;
}

namespace nms_v4 {
using V4BoxEncoding = ov::op::v4::NonMaxSuppression::BoxEncodingType;

struct InfoForNMS4 {
    int64_t max_output_boxes_per_class;
    float iou_threshold;
    float score_threshold;
    float soft_nms_sigma;
    ov::Shape out_shape;
    ov::Shape boxes_shape;
    ov::Shape scores_shape;
    std::vector<float> boxes_data;
    std::vector<float> scores_data;
    size_t out_shape_size;
    bool sort_result_descending;
    ngraph::element::Type output_type;
};

constexpr size_t boxes_port = 0;
constexpr size_t scores_port = 1;

ov::PartialShape infer_selected_indices_shape(const std::vector<std::shared_ptr<ov::HostTensor>>& inputs,
                                              int64_t max_output_boxes_per_class) {
    const auto boxes_ps = inputs[boxes_port]->get_partial_shape();
    const auto scores_ps = inputs[scores_port]->get_partial_shape();

    // NonMaxSuppression produces triplets
    // that have the following format: [batch_index, class_index, box_index]
    ov::PartialShape result = {ov::Dimension::dynamic(), 3};

    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static()) {
        const auto num_boxes_boxes = boxes_ps[1];
        if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static()) {
            const auto num_boxes = num_boxes_boxes.get_length();
            const auto num_classes = scores_ps[1].get_length();

            result[0] = std::min(num_boxes, max_output_boxes_per_class) * num_classes * scores_ps[0].get_length();
        }
    }
    return result;
}

void normalize_corner(float* boxes, const ov::Shape& boxes_shape) {
    size_t total_num_of_boxes = ov::shape_size(boxes_shape) / 4;
    for (size_t i = 0; i < total_num_of_boxes; ++i) {
        float* current_box = boxes + 4 * i;

        float y1 = current_box[0];
        float x1 = current_box[1];
        float y2 = current_box[2];
        float x2 = current_box[3];

        float ymin = std::min(y1, y2);
        float ymax = std::max(y1, y2);
        float xmin = std::min(x1, x2);
        float xmax = std::max(x1, x2);

        current_box[0] = ymin;
        current_box[1] = xmin;
        current_box[2] = ymax;
        current_box[3] = xmax;
    }
}

void normalize_center(float* boxes, const ov::Shape& boxes_shape) {
    size_t total_num_of_boxes = ov::shape_size(boxes_shape) / 4;
    for (size_t i = 0; i < total_num_of_boxes; ++i) {
        float* current_box = boxes + 4 * i;

        float x_center = current_box[0];
        float y_center = current_box[1];
        float width = current_box[2];
        float height = current_box[3];

        float y1 = y_center - height / 2.0f;
        float x1 = x_center - width / 2.0f;
        float y2 = y_center + height / 2.0f;
        float x2 = x_center + width / 2.0f;

        current_box[0] = y1;
        current_box[1] = x1;
        current_box[2] = y2;
        current_box[3] = x2;
    }
}

void normalize_box_encoding(float* boxes, const ov::Shape& boxes_shape, const V4BoxEncoding box_encoding) {
    if (box_encoding == V4BoxEncoding::CORNER) {
        normalize_corner(boxes, boxes_shape);
    } else {
        normalize_center(boxes, boxes_shape);
    }
}

std::vector<float> prepare_boxes_data(const std::shared_ptr<ov::HostTensor>& boxes,
                                      const ov::Shape& boxes_shape,
                                      const V4BoxEncoding box_encoding) {
    auto result = get_floats(boxes, boxes_shape);
    normalize_box_encoding(result.data(), boxes_shape, box_encoding);
    return result;
}

std::vector<float> prepare_scores_data(const std::shared_ptr<ov::HostTensor>& scores, const ov::Shape& scores_shape) {
    auto result = get_floats(scores, scores_shape);
    return result;
}

InfoForNMS4 get_info_for_nms4_eval(const std::shared_ptr<ov::op::v4::NonMaxSuppression>& nms4,
                                   const std::vector<std::shared_ptr<ov::HostTensor>>& inputs) {
    InfoForNMS4 result;

    result.max_output_boxes_per_class = inputs.size() > 2 ? get_integers(inputs[2], ov::Shape({}))[0] : 0;
    result.iou_threshold = inputs.size() > 3 ? get_floats(inputs[3], ov::Shape({}))[0] : 0.0f;
    result.score_threshold = inputs.size() > 4 ? get_floats(inputs[4], ov::Shape({}))[0] : 0.0f;
    result.soft_nms_sigma = inputs.size() > 5 ? get_floats(inputs[5], ov::Shape({}))[0] : 0.0f;

    auto selected_indices_shape = infer_selected_indices_shape(inputs, result.max_output_boxes_per_class);
    result.out_shape = selected_indices_shape.to_shape();

    result.boxes_shape = inputs[boxes_port]->get_shape();
    result.scores_shape = inputs[scores_port]->get_shape();

    result.boxes_data = prepare_boxes_data(inputs[boxes_port], result.boxes_shape, nms4->get_box_encoding());
    result.scores_data = prepare_scores_data(inputs[scores_port], result.scores_shape);

    result.out_shape_size = shape_size(result.out_shape);

    result.sort_result_descending = nms4->get_sort_result_descending();

    result.output_type = nms4->get_output_type();

    return result;
}
}  // namespace nms_v4

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v4::NonMaxSuppression>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    auto info = nms_v4::get_info_for_nms4_eval(op, inputs);

    std::vector<int64_t> selected_indices(info.out_shape_size);
    std::vector<float> selected_scores(info.out_shape_size);
    int64_t valid_outputs = 0;

    ngraph::runtime::reference::non_max_suppression(info.boxes_data.data(),
                                                    info.boxes_shape,
                                                    info.scores_data.data(),
                                                    info.scores_shape,
                                                    info.max_output_boxes_per_class,
                                                    info.iou_threshold,
                                                    info.score_threshold,
                                                    info.soft_nms_sigma,
                                                    selected_indices.data(),
                                                    info.out_shape,
                                                    selected_scores.data(),
                                                    info.out_shape,
                                                    &valid_outputs,
                                                    info.sort_result_descending);

    auto selected_scores_type = (inputs.size() < 4) ? ov::element::f32 : inputs[3]->get_element_type();

    ngraph::runtime::reference::nms_postprocessing(outputs,
                                                   info.output_type,
                                                   selected_indices,
                                                   selected_scores,
                                                   valid_outputs,
                                                   selected_scores_type);
    return true;
}

namespace nms_v3 {
using V3BoxEncoding = ov::op::v3::NonMaxSuppression::BoxEncodingType;

struct InfoForNMS3 {
    int64_t max_output_boxes_per_class;
    float iou_threshold;
    float score_threshold;
    float soft_nms_sigma;
    ov::Shape out_shape;
    ov::Shape boxes_shape;
    ov::Shape scores_shape;
    std::vector<float> boxes_data;
    std::vector<float> scores_data;
    size_t out_shape_size;
    bool sort_result_descending;
    ngraph::element::Type output_type;
};

constexpr size_t boxes_port = 0;
constexpr size_t scores_port = 1;

ov::PartialShape infer_selected_indices_shape(const std::vector<std::shared_ptr<ov::HostTensor>>& inputs,
                                              int64_t max_output_boxes_per_class) {
    const auto boxes_ps = inputs[boxes_port]->get_partial_shape();
    const auto scores_ps = inputs[scores_port]->get_partial_shape();

    // NonMaxSuppression produces triplets
    // that have the following format: [batch_index, class_index, box_index]
    ov::PartialShape result = {ov::Dimension::dynamic(), 3};

    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static()) {
        const auto num_boxes_boxes = boxes_ps[1];
        if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static()) {
            const auto num_boxes = num_boxes_boxes.get_length();
            const auto num_classes = scores_ps[1].get_length();

            result[0] = std::min(num_boxes, max_output_boxes_per_class * num_classes);
        }
    }
    return result;
}

void normalize_corner(float* boxes, const ov::Shape& boxes_shape) {
    size_t total_num_of_boxes = ov::shape_size(boxes_shape) / 4;
    for (size_t i = 0; i < total_num_of_boxes; ++i) {
        float* current_box = boxes + 4 * i;

        float y1 = current_box[0];
        float x1 = current_box[1];
        float y2 = current_box[2];
        float x2 = current_box[3];

        float ymin = std::min(y1, y2);
        float ymax = std::max(y1, y2);
        float xmin = std::min(x1, x2);
        float xmax = std::max(x1, x2);

        current_box[0] = ymin;
        current_box[1] = xmin;
        current_box[2] = ymax;
        current_box[3] = xmax;
    }
}

void normalize_center(float* boxes, const ov::Shape& boxes_shape) {
    size_t total_num_of_boxes = ov::shape_size(boxes_shape) / 4;
    for (size_t i = 0; i < total_num_of_boxes; ++i) {
        float* current_box = boxes + 4 * i;

        float x_center = current_box[0];
        float y_center = current_box[1];
        float width = current_box[2];
        float height = current_box[3];

        float y1 = y_center - height / 2.0f;
        float x1 = x_center - width / 2.0f;
        float y2 = y_center + height / 2.0f;
        float x2 = x_center + width / 2.0f;

        current_box[0] = y1;
        current_box[1] = x1;
        current_box[2] = y2;
        current_box[3] = x2;
    }
}

void normalize_box_encoding(float* boxes, const ov::Shape& boxes_shape, const V3BoxEncoding box_encoding) {
    if (box_encoding == V3BoxEncoding::CORNER) {
        normalize_corner(boxes, boxes_shape);
    } else {
        normalize_center(boxes, boxes_shape);
    }
}

std::vector<float> prepare_boxes_data(const std::shared_ptr<ov::HostTensor>& boxes,
                                      const ov::Shape& boxes_shape,
                                      const V3BoxEncoding box_encoding) {
    auto result = get_floats(boxes, boxes_shape);
    normalize_box_encoding(result.data(), boxes_shape, box_encoding);
    return result;
}

std::vector<float> prepare_scores_data(const std::shared_ptr<ov::HostTensor>& scores, const ov::Shape& scores_shape) {
    auto result = get_floats(scores, scores_shape);
    return result;
}

InfoForNMS3 get_info_for_nms3_eval(const std::shared_ptr<ov::op::v3::NonMaxSuppression>& nms3,
                                   const std::vector<std::shared_ptr<ov::HostTensor>>& inputs) {
    InfoForNMS3 result;

    result.max_output_boxes_per_class = inputs.size() > 2 ? get_integers(inputs[2], ov::Shape({}))[0] : 0;
    result.iou_threshold = inputs.size() > 3 ? get_floats(inputs[3], ov::Shape({}))[0] : 0.0f;
    result.score_threshold = inputs.size() > 4 ? get_floats(inputs[4], ov::Shape({}))[0] : 0.0f;
    result.soft_nms_sigma = inputs.size() > 5 ? get_floats(inputs[5], ov::Shape({}))[0] : 0.0f;

    auto selected_indices_shape = infer_selected_indices_shape(inputs, result.max_output_boxes_per_class);
    result.out_shape = selected_indices_shape.to_shape();

    result.boxes_shape = inputs[boxes_port]->get_shape();
    result.scores_shape = inputs[scores_port]->get_shape();

    result.boxes_data = prepare_boxes_data(inputs[boxes_port], result.boxes_shape, nms3->get_box_encoding());
    result.scores_data = prepare_scores_data(inputs[scores_port], result.scores_shape);

    result.out_shape_size = shape_size(result.out_shape);

    result.sort_result_descending = nms3->get_sort_result_descending();

    result.output_type = nms3->get_output_type();

    return result;
}
}  // namespace nms_v3

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v3::NonMaxSuppression>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    auto info = nms_v3::get_info_for_nms3_eval(op, inputs);

    std::vector<int64_t> selected_indices(info.out_shape_size);
    std::vector<float> selected_scores(info.out_shape_size);
    int64_t valid_outputs = 0;

    ngraph::runtime::reference::non_max_suppression(info.boxes_data.data(),
                                                    info.boxes_shape,
                                                    info.scores_data.data(),
                                                    info.scores_shape,
                                                    info.max_output_boxes_per_class,
                                                    info.iou_threshold,
                                                    info.score_threshold,
                                                    info.soft_nms_sigma,
                                                    selected_indices.data(),
                                                    info.out_shape,
                                                    selected_scores.data(),
                                                    info.out_shape,
                                                    &valid_outputs,
                                                    info.sort_result_descending);

    auto selected_scores_type = (inputs.size() < 4) ? ov::element::f32 : inputs[3]->get_element_type();

    ngraph::runtime::reference::nms_postprocessing(outputs,
                                                   info.output_type,
                                                   selected_indices,
                                                   selected_scores,
                                                   valid_outputs,
                                                   selected_scores_type);
    return true;
}

namespace nms_v1 {
using V1BoxEncoding = ov::op::v1::NonMaxSuppression::BoxEncodingType;

struct InfoForNMS1 {
    int64_t max_output_boxes_per_class;
    float iou_threshold;
    float score_threshold;
    float soft_nms_sigma;
    ov::Shape out_shape;
    ov::Shape boxes_shape;
    ov::Shape scores_shape;
    std::vector<float> boxes_data;
    std::vector<float> scores_data;
    size_t out_shape_size;
    bool sort_result_descending;
    ov::element::Type output_type;
};

constexpr size_t boxes_port = 0;
constexpr size_t scores_port = 1;

ov::PartialShape infer_selected_indices_shape(const std::vector<std::shared_ptr<ov::HostTensor>>& inputs,
                                              int64_t max_output_boxes_per_class) {
    const auto boxes_ps = inputs[boxes_port]->get_partial_shape();
    const auto scores_ps = inputs[scores_port]->get_partial_shape();

    // NonMaxSuppression produces triplets
    // that have the following format: [batch_index, class_index, box_index]
    ov::PartialShape result = {ov::Dimension::dynamic(), 3};

    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static()) {
        const auto num_boxes_boxes = boxes_ps[1];
        if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static()) {
            const auto num_boxes = num_boxes_boxes.get_length();
            const auto num_classes = scores_ps[1].get_length();

            result[0] = std::min(num_boxes, max_output_boxes_per_class * num_classes);
        }
    }
    return result;
}

void normalize_corner(float* boxes, const ov::Shape& boxes_shape) {
    size_t total_num_of_boxes = ov::shape_size(boxes_shape) / 4;
    for (size_t i = 0; i < total_num_of_boxes; ++i) {
        float* current_box = boxes + 4 * i;

        float y1 = current_box[0];
        float x1 = current_box[1];
        float y2 = current_box[2];
        float x2 = current_box[3];

        float ymin = std::min(y1, y2);
        float ymax = std::max(y1, y2);
        float xmin = std::min(x1, x2);
        float xmax = std::max(x1, x2);

        current_box[0] = ymin;
        current_box[1] = xmin;
        current_box[2] = ymax;
        current_box[3] = xmax;
    }
}

void normalize_center(float* boxes, const ov::Shape& boxes_shape) {
    size_t total_num_of_boxes = ov::shape_size(boxes_shape) / 4;
    for (size_t i = 0; i < total_num_of_boxes; ++i) {
        float* current_box = boxes + 4 * i;

        float x_center = current_box[0];
        float y_center = current_box[1];
        float width = current_box[2];
        float height = current_box[3];

        float y1 = y_center - height / 2.0f;
        float x1 = x_center - width / 2.0f;
        float y2 = y_center + height / 2.0f;
        float x2 = x_center + width / 2.0f;

        current_box[0] = y1;
        current_box[1] = x1;
        current_box[2] = y2;
        current_box[3] = x2;
    }
}

void normalize_box_encoding(float* boxes, const ov::Shape& boxes_shape, const V1BoxEncoding box_encoding) {
    if (box_encoding == V1BoxEncoding::CORNER) {
        normalize_corner(boxes, boxes_shape);
    } else {
        normalize_center(boxes, boxes_shape);
    }
}

std::vector<float> prepare_boxes_data(const std::shared_ptr<ov::HostTensor>& boxes,
                                      const ov::Shape& boxes_shape,
                                      const V1BoxEncoding box_encoding) {
    auto result = get_floats(boxes, boxes_shape);
    normalize_box_encoding(result.data(), boxes_shape, box_encoding);
    return result;
}

std::vector<float> prepare_scores_data(const std::shared_ptr<ov::HostTensor>& scores, const ov::Shape& scores_shape) {
    auto result = get_floats(scores, scores_shape);
    return result;
}

InfoForNMS1 get_info_for_nms1_eval(const std::shared_ptr<ov::op::v1::NonMaxSuppression>& nms1,
                                   const std::vector<std::shared_ptr<ov::HostTensor>>& inputs) {
    InfoForNMS1 result;

    result.max_output_boxes_per_class = inputs.size() > 2 ? get_integers(inputs[2], ov::Shape({}))[0] : 0;
    result.iou_threshold = inputs.size() > 3 ? get_floats(inputs[3], ov::Shape({}))[0] : 0.0f;
    result.score_threshold = inputs.size() > 4 ? get_floats(inputs[4], ov::Shape({}))[0] : 0.0f;
    result.soft_nms_sigma = inputs.size() > 5 ? get_floats(inputs[5], ov::Shape({}))[0] : 0.0f;

    auto selected_indices_shape = infer_selected_indices_shape(inputs, result.max_output_boxes_per_class);
    result.out_shape = selected_indices_shape.to_shape();

    result.boxes_shape = inputs[boxes_port]->get_shape();
    result.scores_shape = inputs[scores_port]->get_shape();

    result.boxes_data = prepare_boxes_data(inputs[boxes_port], result.boxes_shape, nms1->get_box_encoding());
    result.scores_data = prepare_scores_data(inputs[scores_port], result.scores_shape);

    result.out_shape_size = shape_size(result.out_shape);

    result.sort_result_descending = nms1->get_sort_result_descending();

    result.output_type = ov::element::i64;

    return result;
}
}  // namespace nms_v1

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v1::NonMaxSuppression>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    auto info = nms_v1::get_info_for_nms1_eval(op, inputs);

    std::vector<int64_t> selected_indices(info.out_shape_size);
    std::vector<float> selected_scores(info.out_shape_size);
    int64_t valid_outputs = 0;

    ngraph::runtime::reference::non_max_suppression(info.boxes_data.data(),
                                                    info.boxes_shape,
                                                    info.scores_data.data(),
                                                    info.scores_shape,
                                                    info.max_output_boxes_per_class,
                                                    info.iou_threshold,
                                                    info.score_threshold,
                                                    info.soft_nms_sigma,
                                                    selected_indices.data(),
                                                    info.out_shape,
                                                    selected_scores.data(),
                                                    info.out_shape,
                                                    &valid_outputs,
                                                    info.sort_result_descending);

    auto selected_scores_type = (inputs.size() < 4) ? ov::element::f32 : inputs[3]->get_element_type();

    ngraph::runtime::reference::nms_postprocessing(outputs,
                                                   info.output_type,
                                                   selected_indices,
                                                   selected_scores,
                                                   valid_outputs,
                                                   selected_scores_type);
    return true;
}