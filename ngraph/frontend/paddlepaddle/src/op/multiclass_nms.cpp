// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multiclass_nms.hpp"
#include <ngraph/opsets/opset6.hpp>

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                NamedOutputs multiclass_nms(const NodeContext& node)
                {
                    auto bboxes = node.get_ng_input("BBoxes");
                    auto scores = node.get_ng_input("Scores");

                    auto score_threshold = node.get_attribute<float>("score_threshold");
                    auto iou_threshold = node.get_attribute<float>("nms_threshold");
                    auto max_output_boxes_per_class = node.get_attribute<int>("nms_top_k");

                    // TODO: dtype, scaler/vector attr, and more strick attributes check
                    auto node_max_output_boxes_per_class = ngraph::opset6::Constant::create<int>(
                        element::i32, Shape{1}, {max_output_boxes_per_class});
                    auto node_iou_threshold = ngraph::opset6::Constant::create<float>(
                        element::f32, Shape{1}, {iou_threshold});
                    auto node_score_threshold = ngraph::opset6::Constant::create<float>(
                        element::f32, Shape{1}, {score_threshold});

                    return node.default_single_output_mapping(
                        {std::make_shared<ngraph::opset6::NonMaxSuppression>(
                            bboxes,
                            scores,
                            node_max_output_boxes_per_class,
                            node_iou_threshold,
                            node_score_threshold)},
                        {"Out"});
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph
