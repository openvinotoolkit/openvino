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

#include <ngraph/opsets/opset6.hpp>
#include "multiclass_nms.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

OutputVector multiclass_nms (const NodeContext& node) {
    auto bboxes = node.get_ng_input("BBoxes");
    auto scores = node.get_ng_input("Scores");

    auto score_threshold = node.get_attribute<float>("score_threshold");
    auto iou_threshold = node.get_attribute<float>("nms_threshold");
    auto max_output_boxes_per_class = node.get_attribute<int>("nms_top_k");

    //TODO: dtype, scaler/vector attr, and more strick attributes check
    auto node_max_output_boxes_per_class = ngraph::opset6::Constant::create<int>(element::i32, Shape{1}, {max_output_boxes_per_class}); 
    auto node_iou_threshold =  ngraph::opset6::Constant::create<float>(element::f32, Shape{1}, {iou_threshold}); 
    auto node_score_threshold = ngraph::opset6::Constant::create<float>(element::f32, Shape{1}, {score_threshold});     

    return {std::make_shared<ngraph::opset6::NonMaxSuppression>(bboxes, scores,
                                    node_max_output_boxes_per_class,
                                    node_iou_threshold,
                                    node_score_threshold)};
}

}}}}