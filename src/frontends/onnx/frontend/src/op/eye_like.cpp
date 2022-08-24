// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/eye_like.hpp"

#include <memory>

#include "exceptions.hpp"
#include "ngraph/output_vector.hpp"
#include "utils/common.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace detail {
namespace {

/// \brief Split a shape returned by a ShapeOf operation into two outputs: width and height.
OutputVector get_shape_width_and_height(const Output<ngraph::Node>& shape) {
    const auto axis = ngraph::op::Constant::create(ngraph::element::i64, {1}, {0});
    const auto height =
        std::make_shared<default_opset::Gather>(shape,
                                                ngraph::op::Constant::create(ngraph::element::i64, {1}, {0}),
                                                axis);
    const auto width =
        std::make_shared<default_opset::Gather>(shape,
                                                ngraph::op::Constant::create(ngraph::element::i64, {1}, {1}),
                                                axis);

    return {width, height};
}

/// \brief Calculate the size of the inner identity matrix and padding values.
/// \param shape Shape of the input tensor returned by a ShapeOf operator.
/// \param k Index of the EyeLike diagonal to be populated with ones.
///          0 populates the main diagonal, k > 0 populates an upper diagonal,
///          and k < 0 populates a lower diagonal.
///
/// \returns A vector of 5 values. The first value is the size of the inner identity matrix.
///          The second value is the padding value for the left side of the inner identity matrix.
///          The third value is the padding value for the right side of the inner identity matrix.
///          The fourth value is the padding value for the top side of the inner identity matrix.
///          The fifth value is the padding value for the bottom side of the inner identity matrix.
OutputVector eyelike_component_dimensions(const Output<ngraph::Node>& shape, std::int64_t k) {
    const auto dims = get_shape_width_and_height(shape);
    const auto width = dims.at(0);
    const auto height = dims.at(1);

    // x1 and y1 are padding values for the left side and top side of the identity matrix.
    const auto x1 = std::max(static_cast<int64_t>(0), k);
    const auto y1 = std::max(static_cast<int64_t>(0), -k);
    const auto x1_const = default_opset::Constant::create(ngraph::element::i64, Shape{1}, {x1});
    const auto y1_const = default_opset::Constant::create(ngraph::element::i64, Shape{1}, {y1});

    // upper_pads is a helper value for calculating the size of the inner identity matrix.
    const auto upper_pads = default_opset::Constant::create(ngraph::element::i64, Shape{2}, {y1, x1});

    // a is the size of the inner identity matrix.
    const auto zero = default_opset::Constant::create(ngraph::element::i64, Shape{1}, {0});
    const auto min_size =
        std::make_shared<default_opset::ReduceMin>(std::make_shared<default_opset::Subtract>(shape, upper_pads),
                                                   zero,
                                                   true);
    const auto a = std::make_shared<default_opset::Maximum>(min_size, zero);

    // x2 and y2 are padding values for the right side and bottom side of the identity matrix.
    // x2 = width - a - x1
    // y2 = height - a - y1
    const auto x2 =
        std::make_shared<default_opset::Subtract>(std::make_shared<default_opset::Subtract>(width, a), x1_const);
    const auto y2 =
        std::make_shared<default_opset::Subtract>(std::make_shared<default_opset::Subtract>(height, a), y1_const);

    return {a, x1_const, x2, y1_const, y2};
}

/// \brief Create a square identity matrix with the specified size and type.
/// \details The identity matrix consists of ones on the main diagonal and zeros elsewhere.
/// \param matrix_size Size of a side of the identity matrix.
/// \param target_type Data type of the identity matrix.
Output<ngraph::Node> square_identity_matrix(const Output<ngraph::Node>& matrix_size, element::Type target_type) {
    // Construct a 1D representation of the identity matrix data
    // One and zero are the values of the identity matrix.
    const auto zero = default_opset::Constant::create(target_type, Shape{1}, {0});
    const auto one = default_opset::Constant::create(target_type, Shape{1}, {1});

    // One row of the identity matrix.
    const auto zeros = std::make_shared<default_opset::Tile>(zero, matrix_size);
    const auto one_followed_by_zeros = std::make_shared<default_opset::Concat>(OutputVector{one, zeros}, 0);

    // The identity matrix as a 1D representation.
    const auto one_int = default_opset::Constant::create(ngraph::element::i64, Shape{1}, {1});
    const auto size_minus_one = std::make_shared<default_opset::Subtract>(matrix_size, one_int);
    const auto one_d_data = std::make_shared<default_opset::Tile>(one_followed_by_zeros, size_minus_one);
    const auto one_d_data_concat = std::make_shared<default_opset::Concat>(OutputVector{one_d_data, one}, 0);

    // Reshape the 1D array to a 2D array
    const auto output_shape = std::make_shared<default_opset::Concat>(OutputVector{matrix_size, matrix_size}, 0);
    const auto diagonal = std::make_shared<default_opset::Reshape>(one_d_data_concat, output_shape, false);
    return diagonal;
}

}  // namespace
}  // namespace detail

namespace set_1 {

OutputVector eye_like(const Node& node) {
    const auto input = node.get_ng_inputs().at(0);

    const auto& input_rank = input.get_partial_shape().rank();
    CHECK_VALID_NODE(node,
                     input_rank.compatible(Rank(2)),
                     "The provided shape rank: ",
                     input_rank.get_length(),
                     " is unsupported, only 2D shapes are supported");

    const auto shift = node.get_attribute_value<std::int64_t>("k", 0);

    std::int64_t dtype;
    element::Type target_type;
    if (node.has_attribute("dtype")) {
        dtype = node.get_attribute_value<std::int64_t>("dtype");
        target_type = common::get_ngraph_element_type(dtype);
    } else {
        target_type = input.get_element_type();
    }

    const auto input_shape = std::make_shared<default_opset::ShapeOf>(input);

    const auto component_dimensions = detail::eyelike_component_dimensions(input_shape, shift);
    const auto identity_matrix = detail::square_identity_matrix(component_dimensions.at(0), target_type);

    const auto pads_begin =
        std::make_shared<default_opset::Concat>(OutputVector{component_dimensions.at(3), component_dimensions.at(1)},
                                                0);
    const auto pads_end =
        std::make_shared<default_opset::Concat>(OutputVector{component_dimensions.at(4), component_dimensions.at(2)},
                                                0);

    const auto zero = default_opset::Constant::create(target_type, Shape{}, {0});
    const auto output =
        std::make_shared<default_opset::Pad>(identity_matrix, pads_begin, pads_end, zero, ov::op::PadMode::CONSTANT);

    return {output};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
