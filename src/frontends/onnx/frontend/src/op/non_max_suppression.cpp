// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/non_max_suppression.hpp"

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/constant.hpp"
#include "utils/reshape.hpp"
using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector non_max_suppression(const ov::frontend::onnx::Node& node) {
    using ov::op::util::is_null;
    // TODO: this op will not be tested until at least
    //       a reference implementation is added

    const auto ng_inputs = node.get_ov_inputs();
    const ov::Output<ov::Node> boxes = ng_inputs.at(0);
    const ov::Output<ov::Node> scores = ng_inputs.at(1);

    ov::Output<ov::Node> max_output_boxes_per_class;
    if (ng_inputs.size() > 2 && !is_null(ng_inputs.at(2))) {
        max_output_boxes_per_class = ov::frontend::onnx::reshape::interpret_as_scalar(ng_inputs.at(2));
    } else {
        max_output_boxes_per_class = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    }

    ov::Output<ov::Node> iou_threshold;
    if (ng_inputs.size() > 3 && !is_null(ng_inputs.at(3))) {
        iou_threshold = ov::frontend::onnx::reshape::interpret_as_scalar(ng_inputs.at(3));
    } else {
        iou_threshold = v0::Constant::create(ov::element::f32, ov::Shape{}, {.0f});
    }

    ov::Output<ov::Node> score_threshold;
    if (ng_inputs.size() > 4 && !is_null(ng_inputs.at(4))) {
        score_threshold = ov::frontend::onnx::reshape::interpret_as_scalar(ng_inputs.at(4));
    } else {
        score_threshold = v0::Constant::create(ov::element::f32, ov::Shape{}, {-std::numeric_limits<float>::max()});
    }

    const auto center_point_box = node.get_attribute_value<std::int64_t>("center_point_box", 0);

    CHECK_VALID_NODE(node,
                     center_point_box == 0 || center_point_box == 1,
                     "Allowed values of the 'center_point_box' attribute are 0 and 1.");

    const auto box_encoding = center_point_box == 0 ? ov::op::v9::NonMaxSuppression::BoxEncodingType::CORNER
                                                    : ov::op::v9::NonMaxSuppression::BoxEncodingType::CENTER;

    return {std::make_shared<v9::NonMaxSuppression>(boxes,
                                                    scores,
                                                    max_output_boxes_per_class,
                                                    iou_threshold,
                                                    score_threshold,
                                                    box_encoding,
                                                    false)};
}

ONNX_OP("NonMaxSuppression", OPSET_SINCE(1), ai_onnx::opset_1::non_max_suppression);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
