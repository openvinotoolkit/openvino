// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <numeric>
#include <memory>

#include <ngraph/opsets/opset7.hpp>
#include "ngraph_ops/multiclass_nms_ie_internal.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/multiclass_nms.hpp"
#include "itt.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::internal::MulticlassNmsIEInternal::type_info;

op::internal::MulticlassNmsIEInternal::MulticlassNmsIEInternal(const Output<Node>& boxes,
                                     const Output<Node>& scores,
                                     const int32_t sort_result_type,
                                     const bool sort_result_across_batch,
                                     const ngraph::element::Type& output_type,
                                     const float iou_threshold,
                                     const float score_threshold,
                                     const int nms_top_k,
                                     const int keep_top_k,
                                     const int background_class,
                                     const float nms_eta)
    : Op({boxes, scores})
    , m_sort_result_type{sort_result_type}
    , m_sort_result_across_batch{sort_result_across_batch}
    , m_output_type{output_type}
    , m_iou_threshold{iou_threshold}
    , m_score_threshold{score_threshold}
    , m_nms_top_k{nms_top_k}
    , m_keep_top_k{keep_top_k}
    , m_background_class{background_class}
    , m_nms_eta{nms_eta} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::internal::MulticlassNmsIEInternal::clone_with_new_inputs(const ngraph::OutputVector &new_args) const {
    INTERNAL_OP_SCOPE(internal_MulticlassNmsIEInternal_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() == 2, "Number of inputs must be 2");

    return std::make_shared<MulticlassNmsIEInternal>(new_args.at(0),
                                                   new_args.at(1),
                                                   m_sort_result_type,
                                                   m_sort_result_across_batch,
                                                   m_output_type,
                                                   m_iou_threshold,
                                                   m_score_threshold,
                                                   m_nms_top_k,
                                                   m_keep_top_k,
                                                   m_background_class,
                                                   m_nms_eta);
}

bool op::internal::MulticlassNmsIEInternal::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(internal_MulticlassNmsIEInternal_visit_attributes);
    visitor.on_attribute("sort_result_type", m_sort_result_type);
    visitor.on_attribute("output_type", m_output_type);
    visitor.on_attribute("nms_top_k", m_nms_top_k);
    visitor.on_attribute("keep_top_k", m_keep_top_k);
    visitor.on_attribute("iou_threshold", m_iou_threshold);
    visitor.on_attribute("score_threshold", m_score_threshold);
    visitor.on_attribute("background_class", m_background_class);
    visitor.on_attribute("nms_eta", m_nms_eta);
    return true;
}

void op::internal::MulticlassNmsIEInternal::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(internal_MulticlassNmsIEInternal_validate_and_infer_types);
    const auto boxes_ps = get_input_partial_shape(0);
    const auto scores_ps = get_input_partial_shape(1);

    auto first_dim_shape = Dimension::dynamic();

    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static()) {
        const auto num_boxes_boxes = boxes_ps[1];
        if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static()) {
            const auto num_boxes = num_boxes_boxes.get_length();
            const auto num_classes = scores_ps[1].get_length();
            int64_t max_output_boxes_per_class = 0;
            if (m_nms_top_k >= 0)
                max_output_boxes_per_class = std::min(num_boxes, (int64_t)m_nms_top_k);
            else
                max_output_boxes_per_class = num_boxes;

            auto max_output_boxes_per_batch = max_output_boxes_per_class * num_classes;
            if (m_keep_top_k >= 0)
                max_output_boxes_per_batch =
                    std::min(max_output_boxes_per_batch, (int64_t)m_keep_top_k);

            first_dim_shape = max_output_boxes_per_batch * scores_ps[0].get_length();
        }
    }

    // 'selected_outputs' have the following format:
    //      [number of selected boxes, [class_id, box_score, xmin, ymin, xmax, ymax]]
    set_output_type(0, element::f32, {first_dim_shape, 6});
    // 'selected_indices' have the following format:
    //      [number of selected boxes, ]
    set_output_type(1, m_output_type, {first_dim_shape, 1});
    // 'selected_num' have the following format:
    //      [num_batches, ]
    if (boxes_ps.rank().is_static() && boxes_ps.rank().get_length() > 0) {
        set_output_type(2, m_output_type, {boxes_ps[0]});
    } else {
        set_output_type(2, m_output_type, {Dimension::dynamic()});
    }
}

// TODO: test usage only
namespace multiclass_nms_v8 {
using SortResultType = op::v8::MulticlassNms::SortResultType;
struct InfoForNMS {
    Shape selected_outputs_shape;
    Shape selected_indices_shape;
    Shape boxes_shape;
    Shape scores_shape;
    std::vector<float> boxes_data;
    std::vector<float> scores_data;
    size_t selected_outputs_shape_size;
    size_t selected_indices_shape_size;
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

PartialShape infer_selected_outputs_shape(const std::vector<std::shared_ptr<HostTensor>>& inputs,
                                    int nms_top_k, int keep_top_k) {
    const auto boxes_ps = inputs[boxes_port]->get_partial_shape();
    const auto scores_ps = inputs[scores_port]->get_partial_shape();

    PartialShape result = {Dimension::dynamic(), 6};

    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static()) {
        const auto num_boxes_boxes = boxes_ps[1];
        if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static()) {
            const auto num_boxes = num_boxes_boxes.get_length();
            const auto num_classes = scores_ps[1].get_length();
            int64_t max_output_boxes_per_class = 0;
            if (nms_top_k >= 0)
                max_output_boxes_per_class = std::min(num_boxes, (int64_t)nms_top_k);
            else
                max_output_boxes_per_class = num_boxes;

            auto max_output_boxes_per_batch = max_output_boxes_per_class * num_classes;
            if (keep_top_k >= 0)
                max_output_boxes_per_batch =
                    std::min(max_output_boxes_per_batch, (int64_t)keep_top_k);

            result[0] = max_output_boxes_per_batch * scores_ps[0].get_length();
        }
    }

    return result;
}

std::vector<float> prepare_boxes_data(const std::shared_ptr<HostTensor>& boxes,
                                        const Shape& boxes_shape) {
    auto result = get_floats(boxes, boxes_shape);
    return result;
}

std::vector<float> prepare_scores_data(const std::shared_ptr<HostTensor>& scores,
                                        const Shape& scores_shape) {
    auto result = get_floats(scores, scores_shape);
    return result;
}

InfoForNMS get_info_for_nms_eval(const op::internal::MulticlassNmsIEInternal* nms,
                                    const std::vector<std::shared_ptr<HostTensor>>& inputs) {
    InfoForNMS result;

    auto selected_outputs_shape =
        infer_selected_outputs_shape(inputs, nms->m_nms_top_k, nms->m_keep_top_k);
    result.selected_outputs_shape = selected_outputs_shape.to_shape();
    result.selected_indices_shape = {result.selected_outputs_shape[0], 1};

    result.boxes_shape = inputs[boxes_port]->get_shape();
    result.scores_shape = inputs[scores_port]->get_shape();

    result.boxes_data = prepare_boxes_data(inputs[boxes_port], result.boxes_shape);
    result.scores_data = prepare_scores_data(inputs[scores_port], result.scores_shape);

    result.selected_outputs_shape_size = shape_size(result.selected_outputs_shape);
    result.selected_indices_shape_size = shape_size(result.selected_indices_shape);

    return result;
}

void multiclass_nms_postprocessing(const HostTensorVector& outputs,
                                    const ngraph::element::Type output_type,
                                    const std::vector<float>& selected_outputs,
                                    const std::vector<int64_t>& selected_indices,
                                    const std::vector<int64_t>& valid_outputs,
                                    const ngraph::element::Type selected_scores_type) {
    auto num_selected = std::accumulate(valid_outputs.begin(), valid_outputs.end(), 0);
    auto partial_shape_0 = outputs[0]->get_partial_shape();
    num_selected = std::min(num_selected, static_cast<int>(partial_shape_0[0].get_max_length()));

    /* shape & type */

    outputs[0]->set_element_type(selected_scores_type); // "selected_outputs"
    //outputs[0]->set_shape(Shape{static_cast<size_t>(num_selected), 6});

    size_t num_of_outputs = outputs.size();

    if (num_of_outputs >= 2) {
        outputs[1]->set_element_type(output_type); // "selected_indices"
        //outputs[1]->set_shape(Shape{static_cast<size_t>(num_selected), 1});
    }

    if (num_of_outputs >= 3) {
        outputs[2]->set_element_type(output_type); // "selected_num"
        //outputs[2]->set_shape(Shape{valid_outputs.size()});
    }

    /* data */
    size_t selected_outputs_size = num_selected * 6;

    switch (selected_scores_type) {
    case element::Type_t::bf16: {
        bfloat16* scores_ptr = outputs[0]->get_data_ptr<bfloat16>();
        for (size_t i = 0; i < selected_outputs_size; ++i) {
            scores_ptr[i] = bfloat16(selected_outputs[i]);
        }
    }
    break;
    case element::Type_t::f16: {
        float16* scores_ptr = outputs[0]->get_data_ptr<float16>();
        for (size_t i = 0; i < selected_outputs_size; ++i) {
            scores_ptr[i] = float16(selected_outputs[i]);
        }
    }
    break;
    case element::Type_t::f32: {
        float* scores_ptr = outputs[0]->get_data_ptr<float>();
        memcpy(
            scores_ptr, selected_outputs.data(), selected_outputs_size * sizeof(float));
    }
    break;
    default:
    break;
    }

    if (num_of_outputs < 2) {
        return;
    }

    size_t selected_indices_size = num_selected * 1;

    if (output_type == ngraph::element::i64) {
        int64_t* indices_ptr = outputs[1]->get_data_ptr<int64_t>();
        memcpy(indices_ptr,
                selected_indices.data(),
                selected_indices_size * sizeof(int64_t));
    } else {
        int32_t* indices_ptr = outputs[1]->get_data_ptr<int32_t>();
        for (size_t i = 0; i < selected_indices_size; ++i) {
            indices_ptr[i] = static_cast<int32_t>(selected_indices[i]);
        }
    }

    if (num_of_outputs < 3) {
        return;
    }

    if (output_type == ngraph::element::i64) {
        int64_t* valid_outputs_ptr = outputs[2]->get_data_ptr<int64_t>();
        memcpy(valid_outputs_ptr,
                valid_outputs.data(),
                valid_outputs.size() * sizeof(int64_t));
    } else {
        int32_t* valid_outputs_ptr = outputs[2]->get_data_ptr<int32_t>();
        for (size_t i = 0; i < valid_outputs.size(); ++i) {
            valid_outputs_ptr[i] = static_cast<int32_t>(valid_outputs[i]);
        }
    }
}
} // namespace multiclass_nms_v8

bool op::internal::MulticlassNmsIEInternal::evaluate(const HostTensorVector& outputs,
                                 const HostTensorVector& inputs) const {
    INTERNAL_OP_SCOPE(internal_MulticlassNmsIEInternal_evaluate);

    auto info = multiclass_nms_v8::get_info_for_nms_eval(this, inputs);

    std::vector<float> selected_outputs(info.selected_outputs_shape_size);
    std::vector<int64_t> selected_indices(info.selected_indices_shape_size);
    std::vector<int64_t> valid_outputs(inputs[0]->get_shape()[0]);

    runtime::reference::multiclass_nms(info.boxes_data.data(),
                                            info.boxes_shape,
                                            info.scores_data.data(),
                                            info.scores_shape,
                                            (ngraph::op::util::NmsBase::SortResultType)m_sort_result_type,
                                            m_sort_result_across_batch,
                                            m_iou_threshold,
                                            m_score_threshold,
                                            m_nms_top_k,
                                            m_keep_top_k,
                                            m_background_class,
                                            m_nms_eta,
                                            selected_outputs.data(),
                                            info.selected_outputs_shape,
                                            selected_indices.data(),
                                            info.selected_indices_shape,
                                            valid_outputs.data());

    auto selected_scores_type = element::f32; // FIXME

    multiclass_nms_v8::multiclass_nms_postprocessing(outputs,
                                            this->m_output_type,
                                            selected_outputs,
                                            selected_indices,
                                            valid_outputs,
                                            selected_scores_type);

    return true;
}