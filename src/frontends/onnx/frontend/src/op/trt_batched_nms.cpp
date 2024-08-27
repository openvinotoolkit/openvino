// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiclass_nms.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/op/concat.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector mmdeploy_trt_batched_nms(const ov::frontend::onnx::Node& node) {
    const auto inputs = node.get_ov_inputs();

    FRONT_END_GENERAL_CHECK(inputs.size() == 2,
                            "The mmdeploy.TRTBatchedNMS operator expects 2 inputs. Got: ",
                            inputs.size());

    const auto& boxes = inputs[0];
    const auto& scores = inputs[1];

    auto boxes_shape = boxes.get_partial_shape();// (N, num_boxes, num_classes（1）, 4)
    auto scores_shape = scores.get_partial_shape();//(N, num_boxes, 1, num_classes).

    //boxes [num_batches, num_boxes, 4]
    //scores [num_batches, num_classes, num_boxes]
    FRONT_END_GENERAL_CHECK((boxes_shape.rank().is_static() && boxes_shape.rank().get_length() == 4 && boxes_shape[2]==1),
                            "The mmdeploy.TRTBatchedNMS operator boxes expects 4 dims and dims[2]==1. Got: ",
                            boxes_shape);


    auto squeeze_axis = std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>({2}));
    auto nms_boxes_inputs = std::make_shared<v0::Squeeze>(boxes, squeeze_axis);

    auto order = std::make_shared<v0::Constant>(element::i32, Shape{3}, std::vector<int32_t>({0,2,1}));
    auto nms_scores_inputs = std::make_shared<v1::Transpose>(scores, order);
    
    auto background_label_id = node.get_attribute_value<int64_t>("background_label_id");
    auto num_classes = node.get_attribute_value<int64_t>("num_classes");

    auto keep_topk = node.get_attribute_value<int64_t>("keep_topk");
    auto topk = node.get_attribute_value<int64_t>("topk");

    auto clip_boxes = node.get_attribute_value<int64_t>("clip_boxes");
    auto is_normalized = node.get_attribute_value<int64_t>("is_normalized");

    auto score_threshold = node.get_attribute_value<float>("score_threshold");
    auto iou_threshold = node.get_attribute_value<float>("iou_threshold");

    v8::MulticlassNms::Attributes attrs;
    attrs.sort_result_type = v8::MulticlassNms::SortResultType::SCORE;
    attrs.sort_result_across_batch = false;
    attrs.output_type = ov::element::i32;
    attrs.iou_threshold = iou_threshold;
    attrs.score_threshold = score_threshold;
    attrs.nms_top_k = topk;
    attrs.keep_top_k = keep_topk;
    attrs.background_class = background_label_id;
    attrs.nms_eta = 1.0;
    attrs.normalized = (is_normalized!=0);

    auto nms_outputs = std::make_shared<v8::MulticlassNms>(nms_boxes_inputs, nms_scores_inputs, attrs)->outputs();

    auto un_squeeze_axis = std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>({0}));    
    auto unsqueeze_output = std::make_shared<v0::Unsqueeze>(nms_outputs[0], un_squeeze_axis)->outputs();

    auto sizes = std::make_shared<v0::Constant>(element::i32, Shape{3}, std::vector<int32_t>({1, 1, 4}));
    auto axis = std::make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>({2}));
    auto split_outputs = std::make_shared<v1::VariadicSplit>(unsqueeze_output[0], axis, sizes)->outputs();


    auto concat_outputs = std::make_shared<v0::Concat>(OutputVector{split_outputs[2], split_outputs[1]}, 2)->outputs();

    return {concat_outputs[0], split_outputs[0]};
}

ONNX_OP("TRTBatchedNMS", OPSET_SINCE(1), ai_onnx::opset_1::mmdeploy_trt_batched_nms, MMDEPLOY_DOMAIN);

}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
