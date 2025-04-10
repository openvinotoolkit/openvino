// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/strided_slice.hpp"
using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector crop(const ov::frontend::onnx::Node& node) {
    // Crop is an obsolete experimental ONNX operation.
    // Crops an image's spatial dimensions.

    const auto inputs = node.get_ov_inputs();
    const auto& input_data = inputs.at(0);

    // Border values: leftBorder, topBorder, rightBorder, bottomBorder.
    const auto border = node.get_attribute_value<std::vector<std::int64_t>>("border");

    std::shared_ptr<ov::Node> end;

    // Set slice begin values to border values (note order of indexes)
    const auto begin =
        v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<std::int64_t>{0, 0, border[1], border[0]});

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
        end = v0::Constant::create(ov::element::i64,
                                   ov::Shape{4},
                                   std::vector<std::int64_t>{0, 0, border[1] + scale[0], border[0] + scale[1]});
    }
    // If scale is not provided, crop the image by values provided in `border`.
    else {
        CHECK_VALID_NODE(node,
                         border.size() == 4,
                         "ONNX Crop expects 4 values in 'border' attribute, found: ",
                         border.size());

        // Calculate ends as shape(input) - border[2:3]
        const auto input_shape = std::make_shared<v3::ShapeOf>(input_data);
        const auto end_offset = v0::Constant::create(ov::element::i64,
                                                     ov::Shape{4},
                                                     std::vector<std::int64_t>{0, 0, -border[3], -border[2]});
        end = std::make_shared<v1::Add>(input_shape, end_offset);
    }

    // Input data shape [N,C,H,W], slicing only along spatial dimensions
    std::vector<int64_t> begin_mask{1, 1, 0, 0};
    std::vector<int64_t> end_mask{1, 1, 0, 0};

    return {std::make_shared<v1::StridedSlice>(input_data, begin, end, begin_mask, end_mask)};
}

ONNX_OP("Crop", OPSET_SINCE(1), ai_onnx::opset_1::crop);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
