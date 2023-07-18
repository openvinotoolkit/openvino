// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/crop.hpp"

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/shape.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector crop(const Node& node) {
    // Crop is an obsolete experimental ONNX operation.
    // Crops an image's spatial dimensions.

    const auto inputs = node.get_ng_inputs();
    const auto& input_data = inputs.at(0);

    // Border values: leftBorder, topBorder, rightBorder, bottomBorder.
    const auto border = node.get_attribute_value<std::vector<std::int64_t>>("border");

    std::shared_ptr<ngraph::Node> end;

    // Set slice begin values to border values (note order of indexes)
    const auto begin = default_opset::Constant::create(ngraph::element::i64,
                                                       Shape{4},
                                                       std::vector<std::int64_t>{0, 0, border[1], border[0]});

    // If scale is given, then start crop at left/top `border`
    // and end on left/top `border` + `scale`.
    if (node.has_attribute("scale")) {
        // List of ints height, width
        const auto scale = node.get_attribute_value<std::vector<std::int64_t>>("scale");

        CHECK_VALID_NODE(node,
                         scale.size() == 2,
                         "ONNX Crop expects 2 values in 'scale' attribute, found: ",
                         scale.size());

        // Set slice end values to topBorder+heightScale and leftBorder+widthScale
        // Note that indexes don't match, e.g. border[0] + scale[1]
        end = default_opset::Constant::create(
            ngraph::element::i64,
            Shape{4},
            std::vector<std::int64_t>{0, 0, border[1] + scale[0], border[0] + scale[1]});
    }
    // If scale is not provided, crop the image by values provided in `border`.
    else {
        CHECK_VALID_NODE(node,
                         border.size() == 4,
                         "ONNX Crop expects 4 values in 'border' attribute, found: ",
                         border.size());

        // Calculate ends as shape(input) - border[2:3]
        const auto input_shape = std::make_shared<default_opset::ShapeOf>(input_data);
        const auto end_offset =
            default_opset::Constant::create(ngraph::element::i64,
                                            Shape{4},
                                            std::vector<std::int64_t>{0, 0, -border[3], -border[2]});
        end = std::make_shared<default_opset::Add>(input_shape, end_offset);
    }

    // Input data shape [N,C,H,W], slicing only along spatial dimensions
    std::vector<int64_t> begin_mask{1, 1, 0, 0};
    std::vector<int64_t> end_mask{1, 1, 0, 0};

    return {std::make_shared<default_opset::StridedSlice>(input_data, begin, end, begin_mask, end_mask)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
