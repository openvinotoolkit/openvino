// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/matrix_nms.hpp"

#include "evaluate_node.hpp"
#include "evaluates_map.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/reference/utils/nms_common.hpp"
#include "openvino/runtime/tensor.hpp"

namespace matrix_nms_v8 {
using SortResultType = ov::op::v8::MatrixNms::SortResultType;
struct InfoForNMS {
    ov::Shape selected_outputs_shape;
    ov::Shape selected_indices_shape;
    ov::Shape boxes_shape;
    ov::Shape scores_shape;
    std::vector<float> boxes_data;
    std::vector<float> scores_data;
    size_t selected_outputs_shape_size;
    size_t selected_indices_shape_size;
};

constexpr size_t boxes_port = 0;
constexpr size_t scores_port = 1;

ov::PartialShape infer_selected_outputs_shape(const ov::TensorVector& inputs, int nms_top_k, int keep_top_k) {
    const auto boxes_ps = inputs[boxes_port].get_shape();
    const auto scores_ps = inputs[scores_port].get_shape();

    ov::PartialShape result = {ov::Dimension::dynamic(), 6};

    if (ov::shape_size(boxes_ps) && ov::shape_size(scores_ps)) {
        const auto num_boxes = boxes_ps[1];
        const auto num_classes = scores_ps[1];
        size_t max_output_boxes_per_class = 0;
        if (nms_top_k >= 0)
            max_output_boxes_per_class = std::min(num_boxes, (size_t)nms_top_k);
        else
            max_output_boxes_per_class = num_boxes;

        auto max_output_boxes_per_batch = max_output_boxes_per_class * num_classes;
        if (keep_top_k >= 0)
            max_output_boxes_per_batch = std::min(max_output_boxes_per_batch, (size_t)keep_top_k);

        result[0] = max_output_boxes_per_batch * scores_ps[0];
    }

    return result;
}

std::vector<float> prepare_boxes_data(const ov::Tensor& boxes, const ov::Shape& boxes_shape) {
    auto result = get_floats(boxes, boxes_shape);
    return result;
}

std::vector<float> prepare_scores_data(const ov::Tensor& scores, const ov::Shape& scores_shape) {
    auto result = get_floats(scores, scores_shape);
    return result;
}

InfoForNMS get_info_for_nms_eval(const std::shared_ptr<ov::op::v8::MatrixNms>& nms, const ov::TensorVector& inputs) {
    InfoForNMS result;
    const auto& nms_attrs = nms->get_attrs();
    const auto nms_top_k = nms_attrs.nms_top_k;
    const auto keep_top_k = nms_attrs.keep_top_k;

    auto selected_outputs_shape = infer_selected_outputs_shape(inputs, nms_top_k, keep_top_k);
    result.selected_outputs_shape = selected_outputs_shape.to_shape();
    result.selected_indices_shape = {result.selected_outputs_shape[0], 1};

    result.boxes_shape = inputs[boxes_port].get_shape();
    result.scores_shape = inputs[scores_port].get_shape();

    result.boxes_data = prepare_boxes_data(inputs[boxes_port], result.boxes_shape);
    result.scores_data = prepare_scores_data(inputs[scores_port], result.scores_shape);

    result.selected_outputs_shape_size = ov::shape_size(result.selected_outputs_shape);
    result.selected_indices_shape_size = ov::shape_size(result.selected_indices_shape);

    return result;
}
}  // namespace matrix_nms_v8

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v8::MatrixNms>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    auto info = matrix_nms_v8::get_info_for_nms_eval(op, inputs);

    std::vector<float> selected_outputs(info.selected_outputs_shape_size);
    std::vector<int64_t> selected_indices(info.selected_indices_shape_size);
    std::vector<int64_t> valid_outputs(info.boxes_shape[0]);

    ov::reference::matrix_nms(info.boxes_data.data(),
                              info.boxes_shape,
                              info.scores_data.data(),
                              info.scores_shape,
                              op->get_attrs(),
                              selected_outputs.data(),
                              info.selected_outputs_shape,
                              selected_indices.data(),
                              info.selected_indices_shape,
                              valid_outputs.data());

    void* pscores = nullptr;
    void* pselected_num = nullptr;
    void* prois;
    size_t num_selected = static_cast<size_t>(std::accumulate(valid_outputs.begin(), valid_outputs.end(), size_t(0)));

    outputs[0].set_shape({num_selected, 6});
    prois = outputs[0].data();

    if (outputs.size() >= 2) {
        outputs[1].set_shape({num_selected, 1});
        pscores = outputs[1].data();
    }
    if (outputs.size() >= 3) {
        pselected_num = outputs[2].data();
    }

    ov::reference::nms_common::nms_common_postprocessing(prois,
                                                         pscores,
                                                         pselected_num,
                                                         op->get_attrs().output_type,
                                                         selected_outputs,
                                                         selected_indices,
                                                         valid_outputs,
                                                         op->get_input_element_type(0));
    return true;
}

template <>
bool evaluate_node<ov::op::v8::MatrixNms>(std::shared_ptr<ov::Node> node,
                                          ov::TensorVector& outputs,
                                          const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v8::MatrixNms>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v8::MatrixNms>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v8::MatrixNms>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v8::MatrixNms>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v8::MatrixNms>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v8::MatrixNms>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v8::MatrixNms>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v8::MatrixNms>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v8::MatrixNms>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v8::MatrixNms>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v8::MatrixNms>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v8::MatrixNms>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v8::MatrixNms>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v8::MatrixNms>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v8::MatrixNms>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v8::MatrixNms>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
