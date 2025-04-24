// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/roi_pooling.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_crop_and_resize_bilinear(const NodeContext& node) {
    default_op_checks(node, 4, {"CropAndResize"});
    auto image = node.get_input(0);
    auto boxes = node.get_input(1);
    auto box_ind = node.get_input(2);
    auto crop_size = node.get_input(3);

    // try to retrive crop_size as the constant, otherwise, we cannot support it
    Shape crop_sizes;
    get_const_input(node, 3, &crop_sizes);

    // concatenate boxes and box_ind inputs because
    // ROIPooling accepts ROIs in a format [batch_id, x_1, y_1, x_2, y_2]
    // prepare box_ind for futher concatenation
    auto const_one = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
    box_ind = make_shared<v0::Unsqueeze>(box_ind, const_one);
    box_ind = make_shared<v0::Convert>(box_ind, element::f32);
    boxes = make_shared<v0::Concat>(OutputVector{box_ind, boxes}, 1);

    // boxes are going in the format [y1, x1, y2, x2]
    // so we need to adjust them to the format [x_1, y_1, x_2, y_2]
    // use Gather operation for the swapping
    auto gather_order = make_shared<v0::Constant>(element::i32, Shape{5}, vector<int32_t>{0, 2, 1, 4, 3});
    auto gather_axis = make_shared<v0::Constant>(element::i32, Shape{1}, vector<int32_t>{1});
    boxes = make_shared<v8::Gather>(boxes, gather_order, gather_axis);

    // prepare input image for ROIPooling
    image = make_transpose(image, {0, 3, 1, 2})->output(0);
    Output<Node> roi_pooling = make_shared<v0::ROIPooling>(image, boxes, crop_sizes, 1.0f, "bilinear");
    roi_pooling = make_transpose(roi_pooling, {0, 2, 3, 1})->output(0);
    set_node_name(node.get_name(), roi_pooling.get_node_shared_ptr());
    return {roi_pooling};
}

OutputVector translate_crop_and_resize_op(const NodeContext& node) {
    // retieve attributes
    // some cases like non-zero extrapolation_value are not supported
    auto method = node.get_attribute<string>("method", "bilinear");
    auto extrapolation_value = node.get_attribute<float>("extrapolation_value", 0.0f);

    TENSORFLOW_OP_VALIDATION(
        node,
        method == "nearest" || method == "bilinear",
        "[TensorFlow Frontend] Inconsistent model: CropAndResize support only bilinear and nearest sampling methods.");

    // TODO 102603: handle a case with non-zero extrapolation_value and nearest sampling
    TENSORFLOW_OP_VALIDATION(
        node,
        method == "bilinear",
        "[TensorFlow Frontend] Internal error: Only CropAndResize with bilinear sampling is supported.");
    TENSORFLOW_OP_VALIDATION(
        node,
        extrapolation_value == 0.0f,
        "[TensorFlow Frontend] Internal error: Only CropAndResize with zero extrapolation_value is supported.");

    return translate_crop_and_resize_bilinear(node);
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
