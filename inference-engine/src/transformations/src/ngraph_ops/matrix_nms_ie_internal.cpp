// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include <ngraph/opsets/opset7.hpp>
#include "ngraph_ops/matrix_nms_ie_internal.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/matrix_nms.hpp"
#include "itt.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::internal::MatrixNmsIEInternal::type_info;

op::internal::MatrixNmsIEInternal::MatrixNmsIEInternal(const Output<Node>& boxes,
                                                               const Output<Node>& scores,
                                                               const Output<Node>& max_output_boxes_per_class,
                                                               const Output<Node>& iou_threshold,
                                                               const Output<Node>& score_threshold,
                                                               int center_point_box,
                                                               bool sort_result_descending,
                                                               const ngraph::element::Type& output_type)
        : Op({boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold}),
          m_center_point_box(center_point_box), m_sort_result_descending(sort_result_descending), m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

op::internal::MatrixNmsIEInternal::MatrixNmsIEInternal(const Output<Node>& boxes,
                                                               const Output<Node>& scores,
                                                               const Output<Node>& max_output_boxes_per_class,
                                                               const Output<Node>& iou_threshold,
                                                               const Output<Node>& score_threshold,
                                                               const Output<Node>& soft_nms_sigma,
                                                               int center_point_box,
                                                               bool sort_result_descending,
                                                               const ngraph::element::Type& output_type)
        : Op({boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, soft_nms_sigma}),
          m_center_point_box(center_point_box), m_sort_result_descending(sort_result_descending), m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::internal::MatrixNmsIEInternal::clone_with_new_inputs(const ngraph::OutputVector &new_args) const {
    INTERNAL_OP_SCOPE(internal_MatrixNmsIEInternal_clone_with_new_inputs);
    if (new_args.size() == 6) {
        return make_shared<MatrixNmsIEInternal>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3),
                                             new_args.at(4), new_args.at(5), m_center_point_box, m_sort_result_descending,
                                             m_output_type);
    } else if (new_args.size() == 5) {
        return make_shared<MatrixNmsIEInternal>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3),
                                             new_args.at(4), m_center_point_box, m_sort_result_descending,
                                             m_output_type);
    }
    throw ngraph::ngraph_error("Unsupported number of inputs: " + std::to_string(new_args.size()));
}

bool op::internal::MatrixNmsIEInternal::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(internal_MatrixNmsIEInternal_visit_attributes);
    visitor.on_attribute("center_point_box", m_center_point_box);
    visitor.on_attribute("sort_result_descending", m_sort_result_descending);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

static constexpr size_t boxes_port = 0;
static constexpr size_t scores_port = 1;
static constexpr size_t max_output_boxes_per_class_port = 2;

int64_t op::internal::MatrixNmsIEInternal::max_boxes_output_from_input() const {
    int64_t max_output_boxes{0};

    size_t num_of_inputs = inputs().size();
    if (num_of_inputs < 3) {
        return 0;
    }

    const auto max_output_boxes_input =
        as_type_ptr<op::Constant>(input_value(max_output_boxes_per_class_port).get_node_shared_ptr());
    max_output_boxes = max_output_boxes_input->cast_vector<int64_t>().at(0);

    return max_output_boxes;
}

void op::internal::MatrixNmsIEInternal::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(internal_MatrixNmsIEInternal_validate_and_infer_types);
    const auto boxes_ps = get_input_partial_shape(boxes_port);
    const auto scores_ps = get_input_partial_shape(scores_port);

    // NonMaxSuppression produces triplets
    // that have the following format: [batch_index, class_index, box_index]
    PartialShape out_shape = {Dimension::dynamic(), 3};

    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static()) {
        const auto num_boxes_boxes = boxes_ps[1];
        const auto max_output_boxes_per_class_node = input_value(max_output_boxes_per_class_port).get_node_shared_ptr();
        if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static() &&
            op::is_constant(max_output_boxes_per_class_node)) {
            const auto num_boxes = num_boxes_boxes.get_length();
            const auto num_classes = scores_ps[1].get_length();
            const auto max_output_boxes_per_class = max_boxes_output_from_input();

            out_shape[0] = std::min(num_boxes, max_output_boxes_per_class) * num_classes *
                           scores_ps[0].get_length();
        }
    }

    set_output_type(0, m_output_type, out_shape);
    set_output_type(1, element::f32, out_shape);
    set_output_type(2, m_output_type, Shape{1});
}

// TODO: test usage only
namespace matrix_nms_v8 {
using V8BoxEncoding = op::v8::MatrixNms::BoxEncodingType;

struct InfoForNMS {
    int64_t max_output_boxes_per_class;
    float iou_threshold;
    float score_threshold;
    float soft_nms_sigma;
    Shape out_shape;
    Shape boxes_shape;
    Shape scores_shape;
    std::vector<float> boxes_data;
    std::vector<float> scores_data;
    size_t out_shape_size;
    bool sort_result_descending;
    ngraph::element::Type output_type;
};

constexpr size_t boxes_port = 0;
constexpr size_t scores_port = 1;

std::vector<float> get_floats(const std::shared_ptr<HostTensor>& input, const Shape& shape) {
    size_t input_size = shape_size(shape);
    std::vector<float> result(input_size);

    switch (input->get_element_type()) {
    case element::Type_t::bf16: {
        bfloat16* p = input->get_data_ptr<bfloat16>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = static_cast<float>(p[i]);
        }
    }
    break;
    case element::Type_t::f16: {
        float16* p = input->get_data_ptr<float16>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = static_cast<float>(p[i]);
        }
    }
    break;
    case element::Type_t::f32: {
        float* p = input->get_data_ptr<float>();
        memcpy(result.data(), p, input_size * sizeof(float));
    }
    break;
    default: throw std::runtime_error("Unsupported data type."); break;
    }

    return result;
}

std::vector<int64_t> get_integers(const std::shared_ptr<HostTensor>& input,
                                    const Shape& shape) {
    size_t input_size = shape_size(shape);
    std::vector<int64_t> result(input_size);

    switch (input->get_element_type()) {
    case element::Type_t::i8: {
        auto p = input->get_data_ptr<int8_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    }
    break;
    case element::Type_t::i16: {
        auto p = input->get_data_ptr<int16_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    }
    break;
    case element::Type_t::i32: {
        auto p = input->get_data_ptr<int32_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    }
    break;
    case element::Type_t::i64: {
        auto p = input->get_data_ptr<int64_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    }
    break;
    case element::Type_t::u8: {
        auto p = input->get_data_ptr<uint8_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    }
    break;
    case element::Type_t::u16: {
        auto p = input->get_data_ptr<uint16_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    }
    break;
    case element::Type_t::u32: {
        auto p = input->get_data_ptr<uint32_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    }
    break;
    case element::Type_t::u64: {
        auto p = input->get_data_ptr<uint64_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    }
    break;
    default:
        throw std::runtime_error("Unsupported data type in op matrix-nms");
        break;
    }

    return result;
}

PartialShape infer_selected_indices_shape(const std::vector<std::shared_ptr<HostTensor>>& inputs,
                                    int64_t max_output_boxes_per_class) {
    const auto boxes_ps = inputs[boxes_port]->get_partial_shape();
    const auto scores_ps = inputs[scores_port]->get_partial_shape();

    // NonMaxSuppression produces triplets
    // that have the following format: [batch_index, class_index, box_index]
    PartialShape result = {Dimension::dynamic(), 3};

    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static()) {
        const auto num_boxes_boxes = boxes_ps[1];
        if (num_boxes_boxes.is_static() && scores_ps[0].is_static() &&
            scores_ps[1].is_static()) {
            const auto num_boxes = num_boxes_boxes.get_length();
            const auto num_classes = scores_ps[1].get_length();

            result[0] = std::min(num_boxes, max_output_boxes_per_class) * num_classes *
                        scores_ps[0].get_length();
        }
    }
    return result;
}

void normalize_corner(float* boxes, const Shape& boxes_shape) {
    size_t total_num_of_boxes = shape_size(boxes_shape) / 4;
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

void normalize_center(float* boxes, const Shape& boxes_shape) {
    size_t total_num_of_boxes = shape_size(boxes_shape) / 4;
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

void normalize_box_encoding(float* boxes,
                            const Shape& boxes_shape,
                            const V8BoxEncoding box_encoding) {
    if (box_encoding == V8BoxEncoding::CORNER) {
        normalize_corner(boxes, boxes_shape);
    } else {
        normalize_center(boxes, boxes_shape);
    }
}

std::vector<float> prepare_boxes_data(const std::shared_ptr<HostTensor>& boxes,
                                        const Shape& boxes_shape,
                                        const V8BoxEncoding box_encoding) {
    auto result = get_floats(boxes, boxes_shape);
    normalize_box_encoding(result.data(), boxes_shape, box_encoding);
    return result;
}

std::vector<float> prepare_scores_data(const std::shared_ptr<HostTensor>& scores,
                                        const Shape& scores_shape) {
    auto result = get_floats(scores, scores_shape);
    return result;
}

InfoForNMS get_info_for_nms_eval(const op::internal::MatrixNmsIEInternal* nms,
                                    const std::vector<std::shared_ptr<HostTensor>>& inputs) {
    InfoForNMS result;

    result.max_output_boxes_per_class =
        inputs.size() > 2 ? get_integers(inputs[2], Shape({}))[0] : 0;
    result.iou_threshold = inputs.size() > 3 ? get_floats(inputs[3], Shape({}))[0] : 0.0f;
    result.score_threshold = inputs.size() > 4 ? get_floats(inputs[4], Shape({}))[0] : 0.0f;
    result.soft_nms_sigma = inputs.size() > 5 ? get_floats(inputs[5], Shape({}))[0] : 0.0f;

    auto selected_indices_shape =
        infer_selected_indices_shape(inputs, result.max_output_boxes_per_class);
    result.out_shape = selected_indices_shape.to_shape();

    result.boxes_shape = inputs[boxes_port]->get_shape();
    result.scores_shape = inputs[scores_port]->get_shape();

    result.boxes_data = prepare_boxes_data(
        inputs[boxes_port], result.boxes_shape, (ngraph::opset7::MatrixNms::BoxEncodingType)nms->m_center_point_box);
    result.scores_data = prepare_scores_data(inputs[scores_port], result.scores_shape);

    result.out_shape_size = shape_size(result.out_shape);

    result.sort_result_descending = nms->m_sort_result_descending;

    result.output_type = nms->m_output_type;

    return result;
}

void matrix_nms_postprocessing(const HostTensorVector& outputs,
                                    const ngraph::element::Type output_type,
                                    const std::vector<int64_t>& selected_indices,
                                    const std::vector<float>& selected_scores,
                                    int64_t valid_outputs,
                                    const ngraph::element::Type selected_scores_type) {
    outputs[0]->set_element_type(output_type);
    //outputs[0]->set_shape(Shape{static_cast<size_t>(valid_outputs), 3});

    size_t num_of_outputs = outputs.size();

    if (num_of_outputs >= 2) {
        outputs[1]->set_element_type(selected_scores_type);
        //outputs[1]->set_shape(Shape{static_cast<size_t>(valid_outputs), 3});
    }

    if (num_of_outputs >= 3) {
        outputs[2]->set_element_type(output_type);
        outputs[2]->set_shape(Shape{1});
    }

    size_t selected_size = valid_outputs * 3;

    if (output_type == ngraph::element::i64) {
        int64_t* indices_ptr = outputs[0]->get_data_ptr<int64_t>();
        memcpy(indices_ptr, selected_indices.data(), selected_size * sizeof(int64_t));
    } else {
        int32_t* indices_ptr = outputs[0]->get_data_ptr<int32_t>();
        for (size_t i = 0; i < selected_size; ++i) {
            indices_ptr[i] = static_cast<int32_t>(selected_indices[i]);
        }
    }

    if (num_of_outputs < 2) {
        return;
    }

    size_t selected_scores_size = selected_scores.size();

    switch (selected_scores_type) {
        case element::Type_t::bf16: {
            bfloat16* scores_ptr = outputs[1]->get_data_ptr<bfloat16>();
            for (size_t i = 0; i < selected_scores_size; ++i) {
                scores_ptr[i] = bfloat16(selected_scores[i]);
            }
        }
        break;
        case element::Type_t::f16: {
            float16* scores_ptr = outputs[1]->get_data_ptr<float16>();
            for (size_t i = 0; i < selected_scores_size; ++i) {
                scores_ptr[i] = float16(selected_scores[i]);
            }
        }
        break;
        case element::Type_t::f32: {
            float* scores_ptr = outputs[1]->get_data_ptr<float>();
            memcpy(scores_ptr, selected_scores.data(), selected_size * sizeof(float));
        }
        break;
        default:
        break;
    }

    if (num_of_outputs < 3) {
        return;
    }

    if (output_type == ngraph::element::i64) {
        int64_t* valid_outputs_ptr = outputs[2]->get_data_ptr<int64_t>();
        *valid_outputs_ptr = valid_outputs;
    } else {
        int32_t* valid_outputs_ptr = outputs[2]->get_data_ptr<int32_t>();
        *valid_outputs_ptr = static_cast<int32_t>(valid_outputs);
    }
}
} // namespace matrix_nms_v8

bool op::internal::MatrixNmsIEInternal::evaluate(const HostTensorVector& outputs,
                                 const HostTensorVector& inputs) const {
    INTERNAL_OP_SCOPE(internal_MatrixNmsIEInternal_evaluate);
    auto info = matrix_nms_v8::get_info_for_nms_eval(this, inputs);
    size_t num_of_outputs = outputs.size();

    std::vector<int64_t> selected_indices(info.out_shape_size);
    std::vector<float> selected_scores(info.out_shape_size);
    int64_t valid_outputs = 0;

    runtime::reference::matrix_nms(info.boxes_data.data(),
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

    auto selected_scores_type =
        (inputs.size() < 4) ? element::f32 : inputs[3]->get_element_type();

    auto out_shape0 = outputs[0]->get_shape();

    matrix_nms_v8::matrix_nms_postprocessing(outputs,
                                            info.output_type,
                                            selected_indices,
                                            selected_scores,
                                            valid_outputs,
                                            selected_scores_type);
    // set default value, make test pass
    for (size_t i = (size_t)valid_outputs * 3; i < info.out_shape_size; i++) {
        if (info.output_type == ngraph::element::i64) {
            outputs[0]->get_data_ptr<int64_t>()[i] = -1;
        } else {
            outputs[0]->get_data_ptr<int32_t>()[i] = -1;
        }
        if (num_of_outputs >= 2) {
            outputs[1]->get_data_ptr<float>()[i] = -1.0f;
        }
    }

    return true;
}