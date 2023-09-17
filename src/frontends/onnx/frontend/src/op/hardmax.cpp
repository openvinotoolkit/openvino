// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/hardmax.hpp"

#include "exceptions.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/validation_util.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector hardmax(const Node& node) {
    const auto input = node.get_ng_inputs().at(0);
    const auto& input_shape = input.get_partial_shape();

    auto axis = node.get_attribute_value<std::int64_t>("axis", 1);
    if (input_shape.rank().is_static()) {
        OPENVINO_SUPPRESS_DEPRECATED_START
        axis = ngraph::normalize_axis(node.get_description(), axis, input_shape.rank());
        OPENVINO_SUPPRESS_DEPRECATED_END
    }

    // reshape to 2D - "batch size" x "input feature dimensions" (NxD)
    const auto coerced_tensor = ngraph::builder::opset1::flatten(input, static_cast<int>(axis));

    const auto coerced_tensor_shape = std::make_shared<default_opset::ShapeOf>(coerced_tensor);
    Output<ngraph::Node> row_size =
        std::make_shared<default_opset::Gather>(coerced_tensor_shape,
                                                default_opset::Constant::create(element::i64, {1}, {1}),
                                                default_opset::Constant::create(element::i64, {}, {0}));
    row_size = ngraph::onnx_import::reshape::interpret_as_scalar(row_size);

    const auto indices_axis = 1;
    const auto topk =
        std::make_shared<default_opset::TopK>(coerced_tensor,
                                              default_opset::Constant::create(ngraph::element::i64, Shape{}, {1}),
                                              indices_axis,
                                              default_opset::TopK::Mode::MAX,
                                              default_opset::TopK::SortType::NONE);

    const auto on_value = default_opset::Constant::create(ngraph::element::i64, Shape{}, {1});
    const auto off_value = default_opset::Constant::create(ngraph::element::i64, Shape{}, {0});

    const auto results =
        std::make_shared<default_opset::OneHot>(topk->output(1), row_size, on_value, off_value, indices_axis);
    const auto converted_results = std::make_shared<default_opset::Convert>(results, input.get_element_type());

    const auto output_shape = std::make_shared<default_opset::ShapeOf>(input);
    return {std::make_shared<default_opset::Reshape>(converted_results, output_shape, false)};
}

}  // namespace set_1
namespace set_13 {
OutputVector hardmax(const Node& node) {
    const auto input = node.get_ng_inputs().at(0);
    const auto& input_shape = input.get_partial_shape();

    auto axis = node.get_attribute_value<std::int64_t>("axis", -1);
    OPENVINO_SUPPRESS_DEPRECATED_START
    axis = ngraph::normalize_axis(node.get_description(), axis, input_shape.rank());
    OPENVINO_SUPPRESS_DEPRECATED_END

    const auto input_runtime_shape = std::make_shared<default_opset::ShapeOf>(input);
    Output<ngraph::Node> row_size =
        std::make_shared<default_opset::Gather>(input_runtime_shape,
                                                default_opset::Constant::create(element::i64, {1}, {axis}),
                                                default_opset::Constant::create(element::i64, {}, {0}));
    row_size = ngraph::onnx_import::reshape::interpret_as_scalar(row_size);

    const auto topk =
        std::make_shared<default_opset::TopK>(input,
                                              default_opset::Constant::create(ngraph::element::i64, Shape{}, {1}),
                                              axis,
                                              default_opset::TopK::Mode::MAX,
                                              default_opset::TopK::SortType::NONE);

    const auto on_value = default_opset::Constant::create(ngraph::element::i64, Shape{}, {1});
    const auto off_value = default_opset::Constant::create(ngraph::element::i64, Shape{}, {0});

    const auto results = std::make_shared<default_opset::OneHot>(topk->output(1), row_size, on_value, off_value, axis);
    const auto converted_results = std::make_shared<default_opset::Convert>(results, input.get_element_type());

    const auto output_shape = std::make_shared<default_opset::ShapeOf>(input);
    return {std::make_shared<default_opset::Reshape>(converted_results, output_shape, false)};
}

}  // namespace set_13
}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
