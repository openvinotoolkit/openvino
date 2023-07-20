// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/non_max_suppression.hpp"

#include <memory>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "onnx_import/core/null_node.hpp"
#include "op/non_max_suppression.hpp"
#include "utils/reshape.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector non_max_suppression(const Node& node) {
    using ngraph::op::is_null;
    // TODO: this op will not be tested until at least
    //       a reference implementation is added

    const auto ng_inputs = node.get_ng_inputs();
    const Output<ngraph::Node> boxes = ng_inputs.at(0);
    const Output<ngraph::Node> scores = ng_inputs.at(1);

    Output<ngraph::Node> max_output_boxes_per_class;
    if (ng_inputs.size() > 2 && !is_null(ng_inputs.at(2))) {
        max_output_boxes_per_class = ngraph::onnx_import::reshape::interpret_as_scalar(ng_inputs.at(2));
    } else {
        max_output_boxes_per_class = default_opset::Constant::create(element::i64, Shape{}, {0});
    }

    Output<ngraph::Node> iou_threshold;
    if (ng_inputs.size() > 3 && !is_null(ng_inputs.at(3))) {
        iou_threshold = ngraph::onnx_import::reshape::interpret_as_scalar(ng_inputs.at(3));
    } else {
        iou_threshold = default_opset::Constant::create(element::f32, Shape{}, {.0f});
    }

    Output<ngraph::Node> score_threshold;
    if (ng_inputs.size() > 4 && !is_null(ng_inputs.at(4))) {
        score_threshold = ngraph::onnx_import::reshape::interpret_as_scalar(ng_inputs.at(4));
    } else {
        score_threshold = default_opset::Constant::create(element::f32, Shape{}, {-std::numeric_limits<float>::max()});
    }

    const auto center_point_box = node.get_attribute_value<std::int64_t>("center_point_box", 0);

    CHECK_VALID_NODE(node,
                     center_point_box == 0 || center_point_box == 1,
                     "Allowed values of the 'center_point_box' attribute are 0 and 1.");

    const auto box_encoding = center_point_box == 0 ? ov::op::v9::NonMaxSuppression::BoxEncodingType::CORNER
                                                    : ov::op::v9::NonMaxSuppression::BoxEncodingType::CENTER;

    return {std::make_shared<ov::op::v9::NonMaxSuppression>(boxes,
                                                            scores,
                                                            max_output_boxes_per_class,
                                                            iou_threshold,
                                                            score_threshold,
                                                            box_encoding,
                                                            false)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
