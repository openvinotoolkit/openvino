//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include <limits>       // std::numeric_limits

#include <ngraph/opsets/opset6.hpp>
#include "yolo_box.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {
    using namespace opset6;
    using namespace element;

NamedOutputs yolo_box (const NodeContext& node_context) {
    auto data = node_context.get_ng_input("X");
    auto image_size = node_context.get_ng_input("ImgSize");

    auto input_shape = data.get_partial_shape();
    int32_t input_height = input_shape[2].get_length();
    int32_t input_width = input_shape[3].get_length();   

    int32_t class_num = node_context.get_attribute<int32_t>("class_num");
    // PDPD anchors attribute is of type int32. Convert to float for computing convinient.
    auto _anchors = node_context.get_attribute<std::vector<int32_t>>("anchors");
    std::vector<float> anchors;
    anchors.resize(_anchors.size());    
    std::transform(_anchors.begin(), _anchors.end(), anchors.begin(), [](int i) {return static_cast<float>(i); });

    int32_t num_anchors = anchors.size()/2; 

    auto default_scale = 1.0f;
    auto scale_x_y = node_context.get_attribute<float>("scale_x_y", default_scale);
    auto downsample_ratio = node_context.get_attribute<int32_t>("downsample_ratio");
    auto input_size = input_height * downsample_ratio;

    auto conf_thresh = node_context.get_attribute<float>("conf_thresh");
    std::vector<float> conf_thresh_mat((float)num_anchors * input_height * input_width, conf_thresh);

    std::vector<int64_t> score_shape {1, input_height * input_width * num_anchors, class_num};

    std::cout << "input_height: " << input_height << " input_width: " << input_width << " input_size: " << input_size<< std::endl;
    std::cout << "num_anchors: " << num_anchors << " scale_x_y: " << scale_x_y << std::endl;
    std::cout << "downsample_ratio: " << downsample_ratio << " conf_thresh: " << conf_thresh << std::endl;
    std::cout << "class_num:  " << class_num << " image_size: " << image_size << std::endl;

    auto clip_bbox = node_context.get_attribute<bool>("clip_bbox"); 

    // main X
    auto node_x_shape = Constant::create<int64_t>(i64, {5}, 
                                                {1, num_anchors, 5 + class_num, input_height, input_width});

    auto node_x_reshape = std::make_shared<Reshape>(data, node_x_shape, false);

    auto node_input_order = Constant::create(i64, {5}, {0, 1, 3, 4, 2});   
    auto node_x_transpose = std::make_shared<Transpose>(node_x_reshape, node_input_order); 

    //  range x/y
    std::vector<float> range_x, range_y;
    for (int32_t i = 0; i < input_width; i++)
    {
        range_x.push_back(i);
    }
    for (int32_t j = 0; j < input_height; j++)
    {
        range_y.push_back(j);
    }
    auto node_range_x = Constant::create<float>(f32, {range_x.size()}, range_x);
    auto node_range_y = Constant::create<float>(f32, {range_y.size()}, range_y);
    
    auto node_range_x_new_shape = Constant::create<int64_t>(i64, {2}, {1, input_width});
    auto node_range_y_new_shape = Constant::create<int64_t>(i64, {2}, {input_height, 1});
                                                 
    auto node_range_x_reshape = std::make_shared<Reshape>(node_range_x, node_range_x_new_shape, false); 
    auto node_range_y_reshape = std::make_shared<Reshape>(node_range_y, node_range_y_new_shape, false); 

    auto node_grid_x = std::make_shared<Tile>(node_range_x_reshape, node_range_y_new_shape);
    auto node_grid_y = std::make_shared<Tile>(node_range_y_reshape, node_range_x_new_shape);

    // main X (part2)
    auto node_split_axis = Constant::create<int64_t>(i64, {1}, {-1});
    auto node_split_lengths = Constant::create<int64_t>(i64, {6}, {1, 1, 1, 1, 1, class_num});
    auto node_split_input = std::make_shared<VariadicSplit>(node_x_transpose, node_split_axis, node_split_lengths);

    auto node_box_x = node_split_input->output(0);
    auto node_box_y = node_split_input->output(1);
    auto node_box_w = node_split_input->output(2);
    auto node_box_h = node_split_input->output(3);
    auto node_conf = node_split_input->output(4);
    auto node_prob = node_split_input->output(5);

    // x/y
    std::shared_ptr<ngraph::Node> node_box_x_sigmoid = std::make_shared<Sigmoid>(node_box_x);
    std::shared_ptr<ngraph::Node> node_box_y_sigmoid = std::make_shared<Sigmoid>(node_box_y);

    if (scale_x_y != default_scale) { //FIXME: float compare
        // TODO
        float bias_x_y = -0.5 * (scale_x_y - 1.0);

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

    auto node_input_h = Constant::create<float>(f32, {1}, {(float)input_height});
    auto node_input_w = Constant::create<float>(f32, {1}, {(float)input_width});

    auto node_box_x_encode = std::make_shared<Divide>(node_box_x_add_grid, node_input_w);    
    auto node_box_y_encode = std::make_shared<Divide>(node_box_y_add_grid, node_input_h); 

    // w/h
    auto node_anchor_tensor = Constant::create<float>(f32, {anchors.size()}, anchors); //FIXME:Paddle2ONNX use float!

    auto node_anchor_shape = Constant::create<int64_t>(i64, {2}, {num_anchors, 2});
    auto node_anchor_tensor_reshape = std::make_shared<Reshape>(node_anchor_tensor, node_anchor_shape, false);

    auto node_input_size = Constant::create<float>(f32, {1}, {(float)input_size});
    auto node_anchors_div_input_size = std::make_shared<Divide>(node_anchor_tensor_reshape, node_input_size);    
  
    auto split_axis = Constant::create<int32_t>(i32, {}, {1});
    auto node_anchor_split = std::make_shared<Split>(node_anchors_div_input_size, split_axis, 2);

    auto node_anchor_w = node_anchor_split->output(0);
    auto node_anchor_h = node_anchor_split->output(1);

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

    auto node_conf_thresh = Constant::create<float>(f32, {conf_thresh_mat.size()}, conf_thresh_mat);
    auto node_conf_shape = Constant::create<int64_t>(i64, {5}, {1, num_anchors, input_height, input_width, 1});
    auto node_conf_thresh_reshape = std::make_shared<Reshape>(node_conf_thresh, node_conf_shape, false);

    auto node_conf_sub = std::make_shared<Subtract>(node_conf_sigmoid, node_conf_thresh_reshape);

    auto node_conf_clip = std::make_shared<Clamp>(node_conf_sub, 0.0f, std::numeric_limits<float>::max()); //FIXME: PDPD not specify min/max

    auto node_zeros = Constant::create<float>(f32, {1}, {0});
    auto node_conf_clip_bool = std::make_shared<Greater>(node_conf_clip, node_zeros);

    auto node_conf_clip_cast = std::make_shared<Convert>(node_conf_clip_bool, f32); //FIMXE: to=1

    auto node_conf_set_zero = std::make_shared<Multiply>(node_conf_sigmoid, node_conf_clip_cast);

    /* probability */
    auto node_prob_sigmoid = std::make_shared<Sigmoid>(node_prob);

    auto node_new_shape = Constant::create<int64_t>(i64, {5}, {1, int(num_anchors), input_height, input_width, 1});
    auto node_conf_new_shape = std::make_shared<Reshape>(node_conf_set_zero, node_new_shape, false);

    // broadcast confidence * probability of each category
    auto node_score = std::make_shared<Multiply>(node_prob_sigmoid, node_conf_new_shape);

    // for bbox which has object (greater than threshold)
    auto node_conf_bool = std::make_shared<Greater>(node_conf_new_shape, node_zeros);

    auto node_box_x_new_shape = std::make_shared<Reshape>(node_box_x_encode, node_new_shape, false);
    auto node_box_y_new_shape = std::make_shared<Reshape>(node_box_y_encode, node_new_shape, false);
    auto node_box_w_new_shape = std::make_shared<Reshape>(node_box_w_encode, node_new_shape, false);
    auto node_box_h_new_shape = std::make_shared<Reshape>(node_box_h_encode, node_new_shape, false);
    auto node_pred_box = std::make_shared<Concat>(OutputVector{node_box_x_new_shape, node_box_y_new_shape, 
                node_box_w_new_shape, node_box_h_new_shape}, 4);

    auto node_conf_cast = std::make_shared<Convert>(node_conf_bool, f32); //FIMXE: to=1

    auto node_pred_box_mul_conf = std::make_shared<Multiply>(node_pred_box, node_conf_cast); //(1,3,19,19,4) (1,3,19,19,1)

    auto node_box_shape = Constant::create<int64_t>(i64, {3}, {1, int(num_anchors) * input_height * input_width, 4});
    auto node_pred_box_new_shape = std::make_shared<Reshape>(node_pred_box_mul_conf, node_box_shape, false); //(1,3*19*19,4)

    auto pred_box_split_axis = Constant::create<int32_t>(i32, {}, {2});
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
    auto squeeze_image_size_axes = Constant::create<int64_t>(i64, {1}, {0});
    auto node_sqeeze_image_size = std::make_shared<Squeeze>(image_size, squeeze_image_size_axes); // input ImgSize

    auto image_size_split_axis = Constant::create<int32_t>(i32, {}, {-1});
    auto node_image_size_split = std::make_shared<Split>(node_sqeeze_image_size, image_size_split_axis, 2);
    auto node_img_height =  node_image_size_split->output(0);
    auto node_img_width =  node_image_size_split->output(1);

    auto node_img_width_cast = std::make_shared<Convert>(node_img_width, f32); //FIMXE: to=1
    auto node_img_height_cast = std::make_shared<Convert>(node_img_height, f32);

    auto node_pred_box_x1_decode = std::make_shared<Multiply>(node_pred_box_x1, node_img_width_cast);
    auto node_pred_box_y1_decode = std::make_shared<Multiply>(node_pred_box_y1, node_img_height_cast);
    auto node_pred_box_x2_decode = std::make_shared<Multiply>(node_pred_box_x2, node_img_width_cast);
    auto node_pred_box_y2_decode = std::make_shared<Multiply>(node_pred_box_y2, node_img_height_cast);

    // reference
    // Paddle/python/paddle/fluid/tests/unittests/test_yolo_box_op.py
    // Paddle/paddle/fluid/operators/detection/yolo_box_op.h
    // Paddle2ONNX/paddle2onnx/op_mapper/detection/yolo_box.py - clip_bbox is not used by Paddle2ONNX.
    std::shared_ptr<ngraph::Node> node_pred_box_result;
    if (clip_bbox) {
        auto node_number_one = Constant::create<float>(f32, {1}, {1.0});
        auto node_new_img_height = std::make_shared<Subtract>(node_img_height_cast, node_number_one);
        auto node_new_img_width = std::make_shared<Subtract>(node_img_width_cast, node_number_one);
        auto node_pred_box_x2_sub_w = std::make_shared<Subtract>(node_pred_box_x2_decode, node_new_img_width); //x2 - (w-1)
        auto node_pred_box_y2_sub_h = std::make_shared<Subtract>(node_pred_box_y2_decode, node_new_img_height); //y2 - (h-1)

        auto max_const = std::numeric_limits<float>::max();
        auto node_pred_box_x1_clip = std::make_shared<Clamp>(node_pred_box_x1_decode, 0.0f, max_const);
        auto node_pred_box_y1_clip = std::make_shared<Clamp>(node_pred_box_y1_decode, 0.0f, max_const);
        auto node_pred_box_x2_clip = std::make_shared<Clamp>(node_pred_box_x2_sub_w, 0.0f, max_const);
        auto node_pred_box_y2_clip = std::make_shared<Clamp>(node_pred_box_y2_sub_h, 0.0f, max_const);

        auto node_pred_box_x2_res = std::make_shared<Subtract>(node_pred_box_x2_decode, node_pred_box_x2_clip);
        auto node_pred_box_y2_res = std::make_shared<Subtract>(node_pred_box_y2_decode, node_pred_box_y2_clip);

        node_pred_box_result = std::make_shared<Concat>(OutputVector{node_pred_box_x1_clip, node_pred_box_y1_clip,
                    node_pred_box_x2_res, node_pred_box_y2_res}, -1); //outputs=node.output('Boxes') 
    }
    else {
        node_pred_box_result = std::make_shared<Concat>(OutputVector{node_pred_box_x1_decode, node_pred_box_y1_decode,
                    node_pred_box_x2_decode, node_pred_box_y2_decode}, -1); //outputs=node.output('Boxes')          
    }

    //
    auto node_score_shape = Constant::create<int64_t>(i64, {score_shape.size()}, score_shape);
    auto node_score_new_shape = std::make_shared<Reshape>(node_score, node_score_shape, false); //outputs=node.output('Scores')

    NamedOutputs outputs;
    outputs["Boxes"] = {node_pred_box_result};
    outputs["Scores"] = {node_score_new_shape};
    return outputs;

}

}}}}