// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <optional>

#include "dimension_util.hpp"
#include "openvino/op/interpolate.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace interpolate {
namespace validate {
/**
 * @brief Validates that input at port number from is 1-D rank.
 *
 * @tparam TShape
 * @param op      Pointer to operator.
 * @param shapes  Vector of op's input shapes.
 * @param port    Port number.
 */
template <class TShape>
void input_rank_1d(const Node* const op, const std::vector<TShape>& shapes, size_t port) {
    constexpr auto exp_rank = 1;
    const auto r = shapes[port].rank();
    NODE_VALIDATION_CHECK(op, r.compatible(exp_rank), "Input [", port, "] is not rank ", exp_rank);
}

/**
 * @brief Validates that inputs from 2nd to last are compatible 1-D rank.
 *
 * @tparam TShape
 * @param op      Pointer to operator.
 * @param shapes  Vector of op's input shapes.
 */
template <class TShape>
void are_inputs_except_first_1d(const Node* const op, const std::vector<TShape>& shapes) {
    for (size_t i = 1; i < shapes.size(); ++i) {
        input_rank_1d(op, shapes, i);
    }
}

/**
 * @brief Check if axes values in range [0,rank].
 *
 * @tparam TContainer  Type of axes container.
 * @param op    Pointer to operator.
 * @param axes  Container with axes to check.
 * @param rank  Maximum value for axes values.
 */
template <class TContainer>
void axes_values(const Node* const op, const TContainer& axes, size_t rank) {
    NODE_VALIDATION_CHECK(op,
                          std::all_of(axes.cbegin(), axes.cend(), ov::cmp::Less<size_t>(rank)),
                          "All axes values should less than input rank: ",
                          rank);
}

/**
 * @brief Check if number of elements in input is same as expected.
 *
 * @param op             Pointer to operator.
 * @param input_name     Input name.
 * @param element_count  Element count in tested input.
 * @param exp_count      Expected element count on tested input.
 */
inline void input_elements_num(const Node* const op,
                               const std::string& input_name,
                               size_t element_count,
                               size_t exp_count) {
    NODE_VALIDATION_CHECK(op,
                          element_count >= exp_count,
                          "The number of elements in the '",
                          input_name,
                          "' input does not match the number of axes ",
                          exp_count);
}
}  // namespace validate

template <class T, class U, typename std::enable_if<std::is_same<T, U>::value>::type* = nullptr>
constexpr bool is_same_instance(const T& lhs, const U& rhs) {
    return std::addressof(lhs) == std::addressof(rhs);
}

template <class T, class U, typename std::enable_if<!std::is_same<T, U>::value>::type* = nullptr>
constexpr bool is_same_instance(const T& lhs, const U& rhs) {
    return false;
}

/**
 * @brief Resize padding to input rank.
 *
 * @tparam TContainer Pads container type.
 * @param op          Pointer to base of interpolate.
 * @param input_rank  Expected padding size.
 * @param pads_begin  Begin padding container.
 * @param pads_end    End padding container.
 */
template <class TContainer>
void resize_padding(const ov::op::util::InterpolateBase* op,
                    size_t input_rank,
                    TContainer& pads_begin,
                    TContainer& pads_end) {
    const auto& op_pads_begin = op->get_attrs().pads_begin;
    const auto& op_pads_end = op->get_attrs().pads_end;

    if (!is_same_instance(op_pads_begin, pads_begin)) {
        pads_begin = TContainer(op_pads_begin.begin(), op_pads_begin.end());
    }

    if (!is_same_instance(op_pads_end, pads_end)) {
        pads_end = TContainer(op_pads_end.begin(), op_pads_end.end());
    }

    pads_begin.resize(input_rank);
    pads_end.resize(input_rank);
}

/**
 * @brief Makes padded shape from input shapes and padding values
 *
 * @note The input shape must be static rank and padding count must match input shape rank.
 *
 * @param input       Input shape used as source for output result.
 * @param pads_begin  Dimensions begin padding values.
 * @param pads_end    Dimensions end padding values.
 * @return TShape     Shape with dimensions of input plus paddings.
 */
template <class TShape, class TInputIter, class TRShape = result_shape_t<TShape>>
TRShape make_padded_shape(const TShape& input, TInputIter pads_begin, TInputIter pads_end) {
    using TDim = typename TShape::value_type;
    TRShape out;
    out.reserve(input.size());
    std::transform(input.cbegin(), input.cend(), std::back_inserter(out), [&pads_begin, &pads_end](const TDim& d) {
        return ov::util::dim::padded(d, (*pads_begin++ + *pads_end++));
    });

    return out;
}

/**
 * @brief Get the axes for interpolate from constant input or default value if op has no axes input.
 *
 * @param op        Pointer to operator.
 * @param port      Axes input port number.
 * @param has_axes  Flag if op has input with axes.
 * @param rank      input shape used for axes values validation.
 * @param ta        Tensor accessor for input data.
 * @return Not null pointer with axes values or null pointer if can't get axes from input.
 */
template <class TShape, class TRes = std::vector<int64_t>>
std::optional<TRes> get_axes(const Node* const op, size_t port, bool has_axes, size_t rank, const ITensorAccessor& ta) {
    std::optional<TRes> axes;
    if (has_axes) {
        using TAxis = typename TRes::value_type;
        axes = std::move(get_input_const_data_as<TShape, TAxis, TRes>(op, port, ta));
        if (axes) {
            validate::axes_values(op, *axes, rank);
        }
    } else {
        axes.emplace(rank);
        std::iota(axes->begin(), axes->end(), 0);
    }
    return axes;
}

/**
 * @brief Set the undefined dimensions on specified axes.
 *
 * @param out   Output shape to update.
 * @param axes  List of axes for update.
 */
template <class TShape, class TContainer>
void set_undefined_dim_on_axes(TShape& out, TContainer& axes) {
    static const auto undefined_dim = Dimension::dynamic();
    for (const auto axis : axes) {
        out[axis] = undefined_dim;
    }
}

/**
 * @brief Update output shape with dimension size from input on specified axes.
 *
 * @param out_shape  Output shape to be updated.
 * @param axes       List of axes for dimension update.
 * @param op         Pointer to operator.
 * @param port       Sizes/output target shape input with values
 * @param ta         Tensor accessor.
 */
template <class TShape, class TContainer>
void update_dims_with_sizes_on_axes(TShape& out_shape,
                                    const TContainer& axes,
                                    const Node* const op,
                                    const size_t port,
                                    const ITensorAccessor& ta) {
    if (const auto sizes = get_input_const_data_as_shape<TShape>(op, port, ta)) {
        validate::input_elements_num(op, "sizes", sizes->size(), axes.size());
        auto sizes_iter = sizes->begin();
        for (const auto axis : axes) {
            out_shape[axis] = *sizes_iter++;
        }
    } else {
        set_undefined_dim_on_axes(out_shape, axes);
    }
}

/**
 * @brief Update output shape by scaling dimensions on axes.
 *
 * @param out_shape  Output shape to update.
 * @param axes       List of axes to scale dimension.
 * @param op         Pointer to operator.
 * @param port       Scales input port number with values.
 * @param ta         Tensor accessor.
 */
template <class TShape>
void update_dims_with_scales_on_axes(TShape& out_shape,
                                     const std::vector<int64_t>& axes,
                                     const Node* const op,
                                     const size_t port,
                                     const ITensorAccessor& ta) {
    if (const auto scales = get_input_const_data_as<TShape, float>(op, port, ta)) {
        validate::input_elements_num(op, "scales", scales->size(), axes.size());
        auto scale_iter = scales->begin();
        for (const auto axis : axes) {
            ov::util::dim::scale(out_shape[axis], *scale_iter++);
        }
    } else {
        set_undefined_dim_on_axes(out_shape, axes);
    }
}
}  // namespace interpolate

namespace v0 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const Interpolate* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& tensor_accessor) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2);
    const auto& img_shape = input_shapes[0];

    auto output_shapes = std::vector<TRShape>(1, img_shape);
    auto& out_shape = output_shapes.front();

    if (img_shape.rank().is_static()) {
        const auto& axes = op->get_attrs().axes;
        const auto img_rank = img_shape.size();

        interpolate::validate::axes_values(op, axes, img_rank);

        if (const auto target_spatial_shape = get_input_const_data_as_shape<TRShape>(op, 1, tensor_accessor)) {
            auto target_spatial_shape_iter = target_spatial_shape->begin();
            for (const auto axis : axes) {
                out_shape[axis] = *target_spatial_shape_iter++;
            }
        } else {
            interpolate::set_undefined_dim_on_axes(out_shape, axes);
        }
    }

    return output_shapes;
}
}  // namespace v0

namespace v4 {
template <class TShape, class TContainer, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const Interpolate* op,
                                 const std::vector<TShape>& input_shapes,
                                 TContainer& pads_begin,
                                 TContainer& pads_end,
                                 const ITensorAccessor& tensor_accessor) {
    const auto has_axes_input = (input_shapes.size() == 4);
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 3 || has_axes_input));

    const auto is_using_scales = (op->get_attrs().shape_calculation_mode == Interpolate::ShapeCalcMode::SCALES);

    interpolate::validate::input_rank_1d(op, input_shapes, is_using_scales ? 2 : 1);

    if (has_axes_input) {
        interpolate::validate::input_rank_1d(op, input_shapes, 3);
    }

    const auto& img_shape = input_shapes[0];
    auto output_shapes = std::vector<TRShape>();

    if (img_shape.rank().is_static()) {
        const auto img_rank = img_shape.size();
        interpolate::resize_padding(op, img_rank, pads_begin, pads_end);

        const auto axes = interpolate::get_axes<TRShape>(op, 3, has_axes_input, img_rank, tensor_accessor);
        if (axes) {
            output_shapes.push_back(interpolate::make_padded_shape(img_shape, pads_begin.cbegin(), pads_end.cbegin()));

            if (is_using_scales) {
                interpolate::update_dims_with_scales_on_axes(output_shapes.front(), *axes, op, 2, tensor_accessor);
            } else {
                interpolate::update_dims_with_sizes_on_axes(output_shapes.front(), *axes, op, 1, tensor_accessor);
            }
        } else {
            output_shapes.push_back(PartialShape::dynamic(img_rank));
        }
    } else {
        output_shapes.push_back(PartialShape::dynamic());
    }
    return output_shapes;
}
}  // namespace v4

namespace v11 {
template <class TShape, class TContainer, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const Interpolate* op,
                                 const std::vector<TShape>& input_shapes,
                                 TContainer& pads_begin,
                                 TContainer& pads_end,
                                 const ITensorAccessor& tensor_accessor) {
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 2 || input_shapes.size() == 3));

    interpolate::validate::are_inputs_except_first_1d(op, input_shapes);

    const auto& img_shape = input_shapes[0];
    auto output_shapes = std::vector<TRShape>();

    if (img_shape.rank().is_static()) {
        const auto img_rank = img_shape.size();
        const auto has_axes_input = (input_shapes.size() == 3);

        interpolate::resize_padding(op, img_rank, pads_begin, pads_end);

        const auto axes = interpolate::get_axes<TRShape>(op, 2, has_axes_input, img_rank, tensor_accessor);
        if (axes) {
            output_shapes.push_back(interpolate::make_padded_shape(img_shape, pads_begin.cbegin(), pads_end.cbegin()));

            if (op->get_attrs().shape_calculation_mode == Interpolate::ShapeCalcMode::SCALES) {
                interpolate::update_dims_with_scales_on_axes(output_shapes.front(), *axes, op, 1, tensor_accessor);
            } else {
                interpolate::update_dims_with_sizes_on_axes(output_shapes.front(), *axes, op, 1, tensor_accessor);
            }
        } else {
            output_shapes.push_back(PartialShape::dynamic(img_rank));
        }
    } else {
        output_shapes.push_back(PartialShape::dynamic());
    }
    return output_shapes;
}
}  // namespace v11
}  // namespace op
}  // namespace ov
