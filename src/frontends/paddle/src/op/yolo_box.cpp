// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <limits>  // std::numeric_limits
#include <numeric>

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
using namespace opset6;
using namespace element;

// reference
// Paddle/python/paddle/fluid/tests/unittests/test_yolo_box_op.py
// Paddle/paddle/fluid/operators/detection/yolo_box_op.h
// Paddle2ONNX/paddle2onnx/op_mapper/detection/yolo_box.py - clip_bbox is not used
// by Paddle2ONNX.
NamedOutputs yolo_box(const NodeContext& node_context) {
    auto data = node_context.get_input("X");
    auto image_size = node_context.get_input("ImgSize");

    // get shape of X
    auto input_shape = std::make_shared<ShapeOf>(data, i64);
    auto indices_batchsize = Constant::create<int32_t>(i64, {1}, {0});
    auto indices_height = Constant::create<int32_t>(i64, {1}, {2});
    auto indices_width = Constant::create<int64_t>(i64, {1}, {3});
    auto const_axis0 = Constant::create<int64_t>(i64, {1}, {0});
    auto input_height = std::make_shared<Gather>(input_shape, indices_height, const_axis0);   // H
    auto input_width = std::make_shared<Gather>(input_shape, indices_width, const_axis0);     // W
    auto batch_size = std::make_shared<Gather>(input_shape, indices_batchsize, const_axis0);  // N

    int32_t class_num = node_context.get_attribute<int32_t>("class_num");
    auto const_class_num = Constant::create<int64_t>(i64, {1}, {class_num});

    // Paddle anchors attribute is of type int32. Convert to float for computing
    // convinient.
    auto _anchors = node_context.get_attribute<std::vector<int32_t>>("anchors");

    std::vector<float> anchors(_anchors.size());
    for (size_t i = 0; i < _anchors.size(); i++)
        anchors[i] = static_cast<float>(_anchors[i]);
    uint32_t num_anchors = static_cast<uint32_t>(anchors.size() / 2);
    auto const_num_anchors = Constant::create<int64_t>(i64, {1}, {num_anchors});

    auto default_scale = 1.0f;
    auto scale_x_y = node_context.get_attribute<float>("scale_x_y", default_scale);

    auto downsample_ratio = node_context.get_attribute<int32_t>("downsample_ratio");
    auto const_downsample_ratio = Constant::create<int64_t>(i64, {1}, {downsample_ratio});
    auto scaled_input_height = std::make_shared<Multiply>(input_height, const_downsample_ratio);
    auto scaled_input_width = std::make_shared<Multiply>(input_width, const_downsample_ratio);

    // score_shape {batch_size, input_height * input_width * num_anchors, class_num}
    auto node_mul_whc = std::make_shared<Multiply>(input_height, input_width);
    node_mul_whc = std::make_shared<Multiply>(node_mul_whc, const_num_anchors);
    auto score_shape = std::make_shared<Concat>(NodeVector{batch_size, node_mul_whc, const_class_num}, 0);

    auto conf_thresh = node_context.get_attribute<float>("conf_thresh");
    auto const_conf_thresh = Constant::create<float>(f32, {1}, {conf_thresh});

    auto clip_bbox = node_context.get_attribute<bool>("clip_bbox");

    // main X
    // node_x_shape {batch_size, num_anchors, 5 + class_num, input_height,
    // input_width}
    auto const_class_num_plus5 = Constant::create<int64_t>(i64, {1}, {5 + class_num});
    auto node_x_shape = std::make_shared<Concat>(
        NodeVector{batch_size, const_num_anchors, const_class_num_plus5, input_height, input_width},
        0);

    auto node_x_reshape = std::make_shared<Reshape>(data, node_x_shape, false);

    auto node_input_order = Constant::create<int64_t>(i64, {5}, {0, 1, 3, 4, 2});
    auto node_x_transpose = std::make_shared<Transpose>(node_x_reshape, node_input_order);

    //  range x/y
    //  range_x: shape {1, input_width} containing 0...input_width
    //  range_y: shape {input_height, 1} containing 0...input_height
    auto const_start = Constant::create<float>(f32, {}, {0.f});
    auto const_step = Constant::create<float>(f32, {}, {1.f});
    auto reduction_axes = Constant::create<int64_t>(i64, {1}, {0});

    auto scaler_input_width = std::make_shared<ReduceMin>(input_width, reduction_axes, false);
    auto range_x = std::make_shared<Range>(const_start, scaler_input_width, const_step, f32);
    auto node_range_x = std::make_shared<Unsqueeze>(range_x, Constant::create<int64_t>(i64, {1}, {0}));

    auto scaler_input_height = std::make_shared<ReduceMin>(input_height, reduction_axes, false);
    auto range_y = std::make_shared<Range>(const_start, scaler_input_height, const_step, f32);
    auto node_range_y = std::make_shared<Unsqueeze>(range_y, Constant::create<int64_t>(i64, {1}, {1}));

    auto node_range_x_shape =
        std::make_shared<Concat>(NodeVector{Constant::create<int64_t>(i64, {1}, {1}), input_width}, 0);
    auto node_range_y_shape =
        std::make_shared<Concat>(NodeVector{input_height, Constant::create<int64_t>(i64, {1}, {1})}, 0);

    auto node_grid_x = std::make_shared<Tile>(node_range_x, node_range_y_shape);  // shape (H, W)
    auto node_grid_y = std::make_shared<Tile>(node_range_y, node_range_x_shape);

    // main X (part2)
    auto node_split_axis = Constant::create<int64_t>(i64, {1}, {-1});
    auto node_split_lengths = Constant::create<int64_t>(i64, {6}, {1, 1, 1, 1, 1, class_num});
    auto node_split_input = std::make_shared<VariadicSplit>(node_x_transpose, node_split_axis, node_split_lengths);

    auto node_box_x = node_split_input->output(0);  // shape (batch_size, num_anchors, H, W, 1)
    auto node_box_y = node_split_input->output(1);
    auto node_box_w = node_split_input->output(2);
    auto node_box_h = node_split_input->output(3);
    auto node_conf = node_split_input->output(4);
    auto node_prob = node_split_input->output(5);

    // x/y
    std::shared_ptr<ov::Node> node_box_x_sigmoid = std::make_shared<Sigmoid>(node_box_x);
    std::shared_ptr<ov::Node> node_box_y_sigmoid = std::make_shared<Sigmoid>(node_box_y);

    if (std::fabs(scale_x_y - default_scale) > 1e-6) {  // float not-equal
        float bias_x_y = -0.5f * (scale_x_y - 1.0f);

        auto scale_x_y_node = Constant::create<float>(f32, {1}, {scale_x_y});
        auto bias_x_y_node = Constant::create<float>(f32, {1}, {bias_x_y});

        node_box_x_sigmoid = std::make_shared<Multiply>(node_box_x_sigmoid, scale_x_y_node);
        node_box_x_sigmoid = std::make_shared<Add>(node_box_x_sigmoid, bias_x_y_node);

        node_box_y_sigmoid = std::make_shared<Multiply>(node_box_y_sigmoid, scale_x_y_node);
        node_box_y_sigmoid = std::make_shared<Add>(node_box_y_sigmoid, bias_x_y_node);
    }

    auto squeeze_box_x = Constant::create<int64_t>(i64, {1}, {4});
    auto node_box_x_squeeze = std::make_shared<Squeeze>(node_box_x_sigmoid, squeeze_box_x);

    auto squeeze_box_y = Constant::create<int64_t>(i64, {1}, {4});
    auto node_box_y_squeeze = std::make_shared<Squeeze>(node_box_y_sigmoid, squeeze_box_y);

    auto node_box_x_add_grid = std::make_shared<Add>(node_grid_x, node_box_x_squeeze);
    auto node_box_y_add_grid = std::make_shared<Add>(node_grid_y, node_box_y_squeeze);

    auto node_input_h = std::make_shared<Convert>(input_height, element::f32);
    auto node_input_w = std::make_shared<Convert>(input_width, element::f32);

    auto node_box_x_encode = std::make_shared<Divide>(node_box_x_add_grid, node_input_w);
    auto node_box_y_encode = std::make_shared<Divide>(node_box_y_add_grid, node_input_h);

    // w/h
    auto node_anchor_tensor = Constant::create<float>(f32, {num_anchors, 2}, anchors);
    auto split_axis = Constant::create<int64_t>(i64, {}, {1});
    auto node_anchor_split = std::make_shared<Split>(node_anchor_tensor, split_axis, 2);

    auto node_anchor_w_origin = node_anchor_split->output(0);
    auto node_anchor_h_origin = node_anchor_split->output(1);

    auto float_input_height = std::make_shared<Convert>(scaled_input_height, element::f32);
    auto node_anchor_h = std::make_shared<Divide>(node_anchor_h_origin, float_input_height);
    auto float_input_width = std::make_shared<Convert>(scaled_input_width, element::f32);
    auto node_anchor_w = std::make_shared<Divide>(node_anchor_w_origin, float_input_width);

    auto node_new_anchor_shape = Constant::create<int64_t>(i64, {4}, {1, num_anchors, 1, 1});
    auto node_anchor_w_reshape = std::make_shared<Reshape>(node_anchor_w, node_new_anchor_shape, false);
    auto node_anchor_h_reshape = std::make_shared<Reshape>(node_anchor_h, node_new_anchor_shape, false);

    auto squeeze_box_wh = Constant::create<int64_t>(i64, {1}, {4});
    auto node_box_w_squeeze = std::make_shared<Squeeze>(node_box_w, squeeze_box_wh);
    auto node_box_h_squeeze = std::make_shared<Squeeze>(node_box_h, squeeze_box_wh);

    auto node_box_w_exp = std::make_shared<Exp>(node_box_w_squeeze);
    auto node_box_h_exp = std::make_shared<Exp>(node_box_h_squeeze);

    auto node_box_w_encode = std::make_shared<Multiply>(node_box_w_exp, node_anchor_w_reshape);
    auto node_box_h_encode = std::make_shared<Multiply>(node_box_h_exp, node_anchor_h_reshape);

    // confidence
    auto node_conf_sigmoid = std::make_shared<Sigmoid>(node_conf);

    auto node_concat = std::make_shared<Concat>(NodeVector{Constant::create<int64_t>(i64, {1}, {1}),
                                                           const_num_anchors,
                                                           input_height,
                                                           input_width,
                                                           Constant::create<int64_t>(i64, {1}, {1})},
                                                0);
    auto node_conf_thresh = std::make_shared<Broadcast>(const_conf_thresh,
                                                        node_concat);  // {1, num_anchors, input_height, input_width, 1}

    auto node_conf_sub = std::make_shared<Subtract>(node_conf_sigmoid, node_conf_thresh);

    auto node_conf_clip = std::make_shared<Clamp>(node_conf_sub, 0.0f, std::numeric_limits<float>::max());

    auto node_zeros = Constant::create<float>(f32, {1}, {0});
    auto node_conf_clip_bool = std::make_shared<Greater>(node_conf_clip, node_zeros);

    auto node_conf_clip_cast = std::make_shared<Convert>(node_conf_clip_bool, f32);

    auto node_conf_set_zero = std::make_shared<Multiply>(node_conf_sigmoid, node_conf_clip_cast);

    /* probability */
    auto node_prob_sigmoid = std::make_shared<Sigmoid>(node_prob);

    auto node_new_shape = std::make_shared<Concat>(
        NodeVector{batch_size, const_num_anchors, input_height, input_width, Constant::create<int64_t>(i64, {1}, {1})},
        0);
    auto node_conf_new_shape =
        std::make_shared<Reshape>(node_conf_set_zero,
                                  node_new_shape,
                                  false);  // {batch_size, int(num_anchors), input_height, input_width, 1}

    // broadcast confidence * probability of each category
    auto node_score = std::make_shared<Multiply>(node_prob_sigmoid, node_conf_new_shape);

    // for bbox which has object (greater than threshold)
    auto node_conf_bool = std::make_shared<Greater>(node_conf_new_shape, node_zeros);

    auto node_box_x_new_shape = std::make_shared<Reshape>(node_box_x_encode, node_new_shape, false);
    auto node_box_y_new_shape = std::make_shared<Reshape>(node_box_y_encode, node_new_shape, false);
    auto node_box_w_new_shape = std::make_shared<Reshape>(node_box_w_encode, node_new_shape, false);
    auto node_box_h_new_shape = std::make_shared<Reshape>(node_box_h_encode, node_new_shape, false);
    auto node_pred_box = std::make_shared<Concat>(
        OutputVector{node_box_x_new_shape, node_box_y_new_shape, node_box_w_new_shape, node_box_h_new_shape},
        4);

    auto node_conf_cast = std::make_shared<Convert>(node_conf_bool, f32);

    auto node_pred_box_mul_conf = std::make_shared<Multiply>(node_pred_box, node_conf_cast);

    auto node_box_shape =
        std::make_shared<Concat>(NodeVector{batch_size, node_mul_whc, Constant::create<int64_t>(i64, {1}, {4})}, 0);
    auto node_pred_box_new_shape =
        std::make_shared<Reshape>(node_pred_box_mul_conf,
                                  node_box_shape,
                                  false);  // {batch_size, int(num_anchors) * input_height * input_width, 4}

    auto pred_box_split_axis = Constant::create<int32_t>(i64, {}, {2});
    auto node_pred_box_split = std::make_shared<Split>(node_pred_box_new_shape, pred_box_split_axis, 4);

    auto node_pred_box_x = node_pred_box_split->output(0);
    auto node_pred_box_y = node_pred_box_split->output(1);
    auto node_pred_box_w = node_pred_box_split->output(2);
    auto node_pred_box_h = node_pred_box_split->output(3);

    /* x,y,w,h -> x1,y1,x2,y2 */
    auto node_number_two = Constant::create<float>(f32, {1}, {2.0f});
    auto node_half_w = std::make_shared<Divide>(node_pred_box_w, node_number_two);
    auto node_half_h = std::make_shared<Divide>(node_pred_box_h, node_number_two);

    auto node_pred_box_x1 = std::make_shared<Subtract>(node_pred_box_x, node_half_w);
    auto node_pred_box_y1 = std::make_shared<Subtract>(node_pred_box_y, node_half_h);

    auto node_pred_box_x2 = std::make_shared<Add>(node_pred_box_x, node_half_w);
    auto node_pred_box_y2 = std::make_shared<Add>(node_pred_box_y, node_half_h);

    /* map normalized coords to original image */
    auto indices_height_imgsize = Constant::create<int32_t>(i64, {1}, {0});
    auto indices_width_imgsize = Constant::create<int64_t>(i64, {1}, {1});
    auto const_axis1 = Constant::create<int64_t>(i64, {1}, {1});
    auto node_img_height =
        std::make_shared<Gather>(image_size, indices_height_imgsize, const_axis1);  // shape_image_size[0]
    auto node_img_width =
        std::make_shared<Gather>(image_size, indices_width_imgsize, const_axis1);  // shape_image_size[1]

    auto node_img_width_cast = std::make_shared<Convert>(node_img_width, f32);
    auto node_img_height_cast = std::make_shared<Convert>(node_img_height, f32);

    auto squeeze_axes2 = Constant::create<int64_t>(i64, {1}, {2});
    auto node_pred_box_x1_reshape =
        std::make_shared<Squeeze>(node_pred_box_x1,
                                  squeeze_axes2);  // shape (N,C,1) -> (N,C) for upcomping multiply.
    auto node_pred_box_y1_reshape = std::make_shared<Squeeze>(node_pred_box_y1, squeeze_axes2);
    auto node_pred_box_x2_reshape = std::make_shared<Squeeze>(node_pred_box_x2, squeeze_axes2);
    auto node_pred_box_y2_reshape = std::make_shared<Squeeze>(node_pred_box_y2, squeeze_axes2);

    auto node_pred_box_x1_squeeze = std::make_shared<Multiply>(node_pred_box_x1_reshape, node_img_width_cast);
    auto node_pred_box_y1_squeeze = std::make_shared<Multiply>(node_pred_box_y1_reshape, node_img_height_cast);
    auto node_pred_box_x2_squeeze = std::make_shared<Multiply>(node_pred_box_x2_reshape, node_img_width_cast);
    auto node_pred_box_y2_squeeze = std::make_shared<Multiply>(node_pred_box_y2_reshape, node_img_height_cast);

    std::shared_ptr<ov::Node> node_pred_box_result;
    if (clip_bbox) {
        auto node_number_one = Constant::create<float>(f32, {1}, {1.0});
        auto node_new_img_height = std::make_shared<Subtract>(node_img_height_cast, node_number_one);
        auto node_new_img_width = std::make_shared<Subtract>(node_img_width_cast, node_number_one);
        auto node_pred_box_x2_sub_w =
            std::make_shared<Subtract>(node_pred_box_x2_squeeze, node_new_img_width);  // x2 - (w-1)
        auto node_pred_box_y2_sub_h =
            std::make_shared<Subtract>(node_pred_box_y2_squeeze, node_new_img_height);  // y2 - (h-1)

        auto max_const = std::numeric_limits<float>::max();
        auto node_pred_box_x1_clip = std::make_shared<Clamp>(node_pred_box_x1_squeeze, 0.0f, max_const);
        auto node_pred_box_y1_clip = std::make_shared<Clamp>(node_pred_box_y1_squeeze, 0.0f, max_const);
        auto node_pred_box_x2_clip = std::make_shared<Clamp>(node_pred_box_x2_sub_w, 0.0f, max_const);
        auto node_pred_box_y2_clip = std::make_shared<Clamp>(node_pred_box_y2_sub_h, 0.0f, max_const);

        auto node_pred_box_x2_res = std::make_shared<Subtract>(node_pred_box_x2_squeeze, node_pred_box_x2_clip);
        auto node_pred_box_y2_res = std::make_shared<Subtract>(node_pred_box_y2_squeeze, node_pred_box_y2_clip);

        auto node_pred_box_x1_clip2 =
            std::make_shared<Unsqueeze>(node_pred_box_x1_clip, squeeze_axes2);  // reshape back to (N,C,1)
        auto node_pred_box_y1_clip2 = std::make_shared<Unsqueeze>(node_pred_box_y1_clip, squeeze_axes2);
        auto node_pred_box_x2_res2 = std::make_shared<Unsqueeze>(node_pred_box_x2_res, squeeze_axes2);
        auto node_pred_box_y2_res2 = std::make_shared<Unsqueeze>(node_pred_box_y2_res, squeeze_axes2);

        node_pred_box_result = std::make_shared<Concat>(
            OutputVector{node_pred_box_x1_clip2, node_pred_box_y1_clip2, node_pred_box_x2_res2, node_pred_box_y2_res2},
            -1);  // outputs=node.output('Boxes')
    } else {
        auto node_pred_box_x1_decode =
            std::make_shared<Unsqueeze>(node_pred_box_x1_squeeze, squeeze_axes2);  // reshape back to (N,C,1)
        auto node_pred_box_y1_decode = std::make_shared<Unsqueeze>(node_pred_box_y1_squeeze, squeeze_axes2);
        auto node_pred_box_x2_decode = std::make_shared<Unsqueeze>(node_pred_box_x2_squeeze, squeeze_axes2);
        auto node_pred_box_y2_decode = std::make_shared<Unsqueeze>(node_pred_box_y2_squeeze, squeeze_axes2);

        node_pred_box_result = std::make_shared<Concat>(OutputVector{node_pred_box_x1_decode,
                                                                     node_pred_box_y1_decode,
                                                                     node_pred_box_x2_decode,
                                                                     node_pred_box_y2_decode},
                                                        -1);  // outputs=node.output('Boxes')
    }

    //
    auto node_score_new_shape =
        std::make_shared<Reshape>(node_score, score_shape, false);  // outputs=node.output('Scores')

    NamedOutputs outputs;
    outputs["Boxes"] = {node_pred_box_result};
    outputs["Scores"] = {node_score_new_shape};
    return outputs;
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
