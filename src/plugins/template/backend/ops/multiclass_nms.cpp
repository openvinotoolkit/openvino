// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/multiclass_nms.hpp"

#include "evaluate_node.hpp"
#include "evaluates_map.hpp"
#include "multiclass_nms_shape_inference.hpp"
#include "openvino/reference/utils/nms_common.hpp"
#include "openvino/runtime/tensor.hpp"

namespace multiclass_nms {
using namespace ov;

struct InfoForNMS {
    Shape selected_outputs_shape;
    Shape selected_indices_shape;
    Shape selected_numrois_shape;
    Shape boxes_shape;
    Shape scores_shape;
    Shape roisnum_shape;
    std::vector<float> boxes_data;
    std::vector<float> scores_data;
    std::vector<int64_t> roisnum_data;
    size_t selected_outputs_shape_size;
    size_t selected_indices_shape_size;
    size_t selected_numrois_shape_size;
};

static std::vector<float> prepare_boxes_data(const ov::Tensor& boxes, const Shape& boxes_shape) {
    auto result = get_floats(boxes, boxes_shape);
    return result;
}

static std::vector<float> prepare_scores_data(const ov::Tensor& scores, const Shape& scores_shape) {
    auto result = get_floats(scores, scores_shape);
    return result;
}

static std::vector<int64_t> prepare_roisnum_data(const ov::Tensor& roisnum, const Shape& roisnum_shape) {
    auto result = get_integers(roisnum, roisnum_shape);
    return result;
}

constexpr size_t boxes_port = 0;
constexpr size_t scores_port = 1;
constexpr size_t roisnum_port = 2;

InfoForNMS get_info_for_nms_eval(const std::shared_ptr<op::util::MulticlassNmsBase>& nms,
                                 const ov::TensorVector& inputs) {
    InfoForNMS result;

    const auto boxes_ps = inputs[boxes_port].get_shape();
    const auto scores_ps = inputs[scores_port].get_shape();
    std::vector<PartialShape> input_shapes = {boxes_ps, scores_ps};
    if (nms->get_input_size() == 3) {
        const auto roisnum_ps = inputs[roisnum_port].get_shape();
        input_shapes.push_back(roisnum_ps);
    }

    const auto output_shapes = ov::op::shape_infer(nms.get(), input_shapes);

    result.selected_outputs_shape = output_shapes[0].get_max_shape();
    result.selected_indices_shape = output_shapes[1].get_max_shape();
    result.selected_numrois_shape = output_shapes[2].to_shape();

    result.boxes_shape = inputs[boxes_port].get_shape();
    result.scores_shape = inputs[scores_port].get_shape();

    result.boxes_data = prepare_boxes_data(inputs[boxes_port], result.boxes_shape);
    result.scores_data = prepare_scores_data(inputs[scores_port], result.scores_shape);

    if (inputs.size() == 3) {
        result.roisnum_shape = inputs[roisnum_port].get_shape();
        result.roisnum_data = prepare_roisnum_data(inputs[roisnum_port], result.roisnum_shape);
    }

    result.selected_outputs_shape_size = shape_size(result.selected_outputs_shape);
    result.selected_indices_shape_size = shape_size(result.selected_indices_shape);
    result.selected_numrois_shape_size = shape_size(result.selected_numrois_shape);

    return result;
}
}  // namespace multiclass_nms

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v8::MulticlassNms>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    auto info = multiclass_nms::get_info_for_nms_eval(op, inputs);

    std::vector<float> selected_outputs(info.selected_outputs_shape_size);
    std::vector<int64_t> selected_indices(info.selected_indices_shape_size);
    std::vector<int64_t> valid_outputs(info.selected_numrois_shape_size);

    ov::reference::multiclass_nms(info.boxes_data.data(),
                                  info.boxes_shape,
                                  info.scores_data.data(),
                                  info.scores_shape,
                                  nullptr,
                                  ov::Shape(),  // won't be used
                                  op->get_attrs(),
                                  selected_outputs.data(),
                                  info.selected_outputs_shape,
                                  selected_indices.data(),
                                  info.selected_indices_shape,
                                  valid_outputs.data());

    void* pscores = nullptr;
    void* pselected_num = nullptr;
    void* prois;
    size_t num_selected = static_cast<size_t>(std::accumulate(valid_outputs.begin(), valid_outputs.end(), int64_t(0)));

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

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v9::MulticlassNms>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    auto info = multiclass_nms::get_info_for_nms_eval(op, inputs);

    std::vector<float> selected_outputs(info.selected_outputs_shape_size);
    std::vector<int64_t> selected_indices(info.selected_indices_shape_size);
    std::vector<int64_t> valid_outputs(info.selected_numrois_shape_size);

    ov::reference::multiclass_nms(info.boxes_data.data(),
                                  info.boxes_shape,
                                  info.scores_data.data(),
                                  info.scores_shape,
                                  info.roisnum_data.data(),
                                  info.roisnum_shape,
                                  op->get_attrs(),
                                  selected_outputs.data(),
                                  info.selected_outputs_shape,
                                  selected_indices.data(),
                                  info.selected_indices_shape,
                                  valid_outputs.data());

    void* pscores = nullptr;
    void* pselected_num = nullptr;
    void* prois;
    size_t num_selected = static_cast<size_t>(std::accumulate(valid_outputs.begin(), valid_outputs.end(), 0));

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
bool evaluate_node<ov::op::v8::MulticlassNms>(std::shared_ptr<ov::Node> node,
                                              ov::TensorVector& outputs,
                                              const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v8::MulticlassNms>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v8::MulticlassNms>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v8::MulticlassNms>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v8::MulticlassNms>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v8::MulticlassNms>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v8::MulticlassNms>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v8::MulticlassNms>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v8::MulticlassNms>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v8::MulticlassNms>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v8::MulticlassNms>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v8::MulticlassNms>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v8::MulticlassNms>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v8::MulticlassNms>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v8::MulticlassNms>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v8::MulticlassNms>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v8::MulticlassNms>(node), outputs, inputs);
    default:
        OPENVINO_THROW(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                       std::string("in evaluate_node()"));
    }
}

template <>
bool evaluate_node<ov::op::v9::MulticlassNms>(std::shared_ptr<ov::Node> node,
                                              ov::TensorVector& outputs,
                                              const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v9::MulticlassNms>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v9::MulticlassNms>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v9::MulticlassNms>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v9::MulticlassNms>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v9::MulticlassNms>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v9::MulticlassNms>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v9::MulticlassNms>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v9::MulticlassNms>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v9::MulticlassNms>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v9::MulticlassNms>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v9::MulticlassNms>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v9::MulticlassNms>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v9::MulticlassNms>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v9::MulticlassNms>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v9::MulticlassNms>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v9::MulticlassNms>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
