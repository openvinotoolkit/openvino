// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "compare.hpp"
#include "dimension_util.hpp"
#include "openvino/op/nms_rotated.hpp"
#include "openvino/op/non_max_suppression.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace nms {
namespace validate {

template <class TShape>
bool scalar(const TShape& shape) {
    return shape.compatible(result_shape_t<TShape>{});
}

template <class TShape>
bool scalar_or_1d_tensor_with_1_element(const TShape& shape) {
    return scalar(shape) || shape.compatible(result_shape_t<TShape>{1});
}

template <class TShape>
void boxes_shape(const Node* const op, const std::vector<TShape>& input_shapes) {
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           input_shapes[0].rank().compatible(3),
                           "Expected a 3D tensor for the 'boxes' input");
}

template <class TShape>
void scores_shape(const Node* const op, const std::vector<TShape>& input_shapes) {
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           input_shapes[1].rank().compatible(3),
                           "Expected a 3D tensor for the 'scores' input");
}

template <class TShape>
void num_batches(const Node* const op, const std::vector<TShape>& input_shapes) {
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           input_shapes[0][0].compatible(input_shapes[1][0]),
                           "The first dimension of both 'boxes' and 'scores' must match.");
}

template <class TShape>
void num_boxes(const Node* const op, const std::vector<TShape>& input_shapes) {
    NODE_SHAPE_INFER_CHECK(
        op,
        input_shapes,
        input_shapes[0][1].compatible(input_shapes[1][2]),
        "'boxes' and 'scores' input shapes must match at the second and third dimension respectively. Boxes: ");
}

template <class TShape>
void boxes_last_dim(const Node* const op, const std::vector<TShape>& input_shapes) {
    using TDim = typename TShape::value_type;
    TDim box_def_size = ov::is_type<v13::NMSRotated>(op) ? 5 : 4;
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           input_shapes[0][2].compatible(box_def_size),
                           "The last dimension of the 'boxes' input must be equal to ",
                           box_def_size);
}

template <class T>
void shapes(const Node* op, const std::vector<T>& input_shapes) {
    const auto inputs_size = input_shapes.size();
    NODE_VALIDATION_CHECK(op, cmp::Between<size_t>(1, 6)(inputs_size));

    nms::validate::boxes_shape(op, input_shapes);
    nms::validate::scores_shape(op, input_shapes);
    if (inputs_size > 2) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               nms::validate::scalar(input_shapes[2]),
                               "Expected a scalar for the 'max_output_boxes_per_class' input.");
    }

    if (inputs_size > 3) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               nms::validate::scalar(input_shapes[3]),
                               "Expected a scalar for the 'iou_threshold' input");
    }

    if (inputs_size > 4) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               nms::validate::scalar(input_shapes[4]),
                               "Expected a scalar for the 'score_threshold' input");
    }

    const auto& boxes_shape = input_shapes[0];
    const auto& scores_shape = input_shapes[1];

    if (boxes_shape.rank().is_static()) {
        if (scores_shape.rank().is_static()) {
            nms::validate::num_batches(op, input_shapes);
            nms::validate::num_boxes(op, input_shapes);
        }
        nms::validate::boxes_last_dim(op, input_shapes);
    }
}
}  // namespace validate

template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const Node* op,
                                 const std::vector<T>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    using TDim = typename T::value_type;
    using namespace ov::util;

    const auto& boxes_shape = input_shapes[0];
    const auto& scores_shape = input_shapes[1];

    auto output_shapes = std::vector<TRShape>{TRShape{TDim(dim::inf_bound), 3}};
    if (boxes_shape.rank().is_static()) {
        const auto max_out_boxes_per_class = get_input_const_data_as<TRShape, int64_t>(op, 2, ta);
        auto max_out_class_boxes =
            max_out_boxes_per_class ? TDim(max_out_boxes_per_class->front()) : TDim(dim::inf_bound);

        if (scores_shape.rank().is_static()) {
            max_out_class_boxes *= scores_shape[1];
        }

        const auto& num_boxes = boxes_shape[1];
        if (num_boxes.is_static() && max_out_class_boxes.is_static()) {
            auto& selected_boxes = output_shapes[0][0];
            selected_boxes = std::min(num_boxes.get_length(), max_out_class_boxes.get_length());
        }
    }

    return output_shapes;
}

// Shape inference for NMS operators which can force static output using PartialShape
template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const Node* op,
                                 const std::vector<T>& input_shapes,
                                 const ITensorAccessor& ta,
                                 const bool static_output) {
    // Note: static_output parameter of this shape_infer is exclusively made for GPU internal needs
    // To be removed after GPU supports dynamic NMS
    const auto inputs_size = input_shapes.size();
    NODE_VALIDATION_CHECK(op, cmp::Between<size_t>(1, 7)(inputs_size));
    using TDim = typename TRShape::value_type;
    using V = typename TDim::value_type;
    using namespace ov::util;

    nms::validate::boxes_shape(op, input_shapes);
    nms::validate::scores_shape(op, input_shapes);

    if (inputs_size > 2) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               nms::validate::scalar_or_1d_tensor_with_1_element(input_shapes[2]),
                               "Expected 0D or 1D tensor for the 'max_output_boxes_per_class' input");
    }

    if (inputs_size > 3) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               nms::validate::scalar_or_1d_tensor_with_1_element(input_shapes[3]),
                               "Expected 0D or 1D tensor for the 'iou_threshold' input");
    }

    if (inputs_size > 4) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               nms::validate::scalar_or_1d_tensor_with_1_element(input_shapes[4]),
                               "Expected 0D or 1D tensor for the 'score_threshold' input");
    }

    if (inputs_size > 5) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               nms::validate::scalar_or_1d_tensor_with_1_element(input_shapes[5]),
                               "Expected 0D or 1D tensor for the 'soft_nms_sigma' input");
    }

    const auto& boxes_shape = input_shapes[0];

    auto out_shape = TRShape{TDim(dim::inf_bound), 3};
    if (boxes_shape.rank().is_static()) {
        const auto& scores_shape = input_shapes[1];

        if (scores_shape.rank().is_static()) {
            nms::validate::num_batches(op, input_shapes);
            nms::validate::num_boxes(op, input_shapes);

            auto& selected_boxes = out_shape[0];
            if (const auto max_out_boxes_per_class = get_input_const_data_as<TRShape, int64_t>(op, 2, ta)) {
                const auto& num_boxes = boxes_shape[1];
                const auto min_selected_boxes =
                    std::min(num_boxes.get_max_length(), static_cast<V>(max_out_boxes_per_class->front()));
                selected_boxes = static_output ? TDim{min_selected_boxes} : TDim{0, min_selected_boxes};
            }

            selected_boxes *= scores_shape[0].get_max_length();
            selected_boxes *= scores_shape[1].get_max_length();
        }

        nms::validate::boxes_last_dim(op, input_shapes);
    }

    auto output_shapes = std::vector<TRShape>(2, out_shape);
    output_shapes.emplace_back(std::initializer_list<V>{1});
    return output_shapes;
}
}  // namespace nms

namespace v1 {

template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const NonMaxSuppression* op,
                                 const std::vector<T>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    nms::validate::shapes(op, input_shapes);
    return nms::shape_infer(op, input_shapes, ta);
}
}  // namespace v1

namespace v3 {

template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const NonMaxSuppression* op,
                                 const std::vector<T>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    nms::validate::shapes(op, input_shapes);
    return nms::shape_infer(op, input_shapes, ta);
}
}  // namespace v3
namespace v4 {

template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const NonMaxSuppression* op,
                                 const std::vector<T>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    using TDim = typename TRShape::value_type;
    using V = typename TDim::value_type;
    using namespace ov::util;
    nms::validate::shapes(op, input_shapes);

    const auto& boxes_shape = input_shapes[0];
    const auto& scores_shape = input_shapes[1];

    auto output_shapes = std::vector<TRShape>{TRShape{TDim(dim::inf_bound), 3}};

    if (boxes_shape.rank().is_static() && scores_shape.rank().is_static()) {
        const auto& num_boxes = boxes_shape[1];
        if (num_boxes.is_static()) {
            if (const auto max_out_boxes_per_class = get_input_const_data_as<TRShape, int64_t>(op, 2, ta)) {
                auto& selected_boxes = output_shapes[0][0];
                selected_boxes = std::min(num_boxes.get_length(), static_cast<V>(max_out_boxes_per_class->front()));
                selected_boxes *= scores_shape[0].get_max_length();
                selected_boxes *= scores_shape[1].get_max_length();
            }
        }
    }

    return output_shapes;
}
}  // namespace v4

namespace v5 {

template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const NonMaxSuppression* op,
                                 const std::vector<T>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor(),
                                 const bool static_output = !std::is_same<T, PartialShape>::value) {
    return nms::shape_infer(op, input_shapes, ta, static_output);
}
}  // namespace v5

namespace v9 {

template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const NonMaxSuppression* op,
                                 const std::vector<T>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor(),
                                 const bool static_output = !std::is_same<T, PartialShape>::value) {
    return nms::shape_infer(op, input_shapes, ta, static_output);
}
}  // namespace v9

namespace v13 {
template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const NMSRotated* op,
                                 const std::vector<T>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor(),
                                 const bool static_output = !std::is_same<T, PartialShape>::value) {
    return nms::shape_infer(op, input_shapes, ta, static_output);
}
}  // namespace v13
}  // namespace op
}  // namespace ov
