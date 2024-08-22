// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/opsets/opset8.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {

ov::OutputVector trt_batched_nms(const ov::frontend::onnx::Node& node) {

    auto background_label_id = node.get_attribute_value<int>("background_label_id");
    auto num_classes = node.get_attribute_value<int>("num_classes");

    auto keep_topk = node.get_attribute_value<int>("keep_topk");
    auto topk = node.get_attribute_value<int>("topk");

    // auto return_index = node.get_attribute_value<bool>("return_index");
    auto clip_boxes = node.get_attribute_value<bool>("clip_boxes");
    auto is_normalized = node.get_attribute_value<bool>("is_normalized");

    auto score_threshold = node.get_attribute_value<float>("score_threshold");
    auto iou_threshold = node.get_attribute_value<float>("iou_threshold");

    auto boxes = node.get_ov_inputs().at(0);
    auto scores = node.get_ov_inputs().at(0);

    CHECK_VALID_NODE(node, node.get_outputs_size() != 2,
                     "Unexpected number of outputs of MatrixNMS." + std::to_string(node.get_outputs_size()));


    // NamedOutputs named_outputs;
    // std::vector<Output<Node>> nms_outputs;
    opset8::MatrixNms::Attributes attrs;
    attrs.nms_top_k = topk;
    // attrs.post_threshold = iou_threshold;
    attrs.score_threshold = score_threshold;
    attrs.sort_result_type = opset8::MatrixNms::SortResultType::CLASSID;
    attrs.keep_top_k = keep_topk;
    attrs.background_class = background_label_id;
    attrs.normalized = is_normalized;
    attrs.output_type = ov::element::i64;
    attrs.sort_result_across_batch = true;
    attrs.decay_function = opset8::MatrixNms::DecayFunction::LINEAR;;
    attrs.gaussian_sigma = 0;
    
    auto nms_outputs = std::make_shared<opset8::MatrixNms>(boxes, scores, attrs)->outputs();

    // named_outputs["dets"] = {nms_outputs[0]};
    // named_outputs["labels"] = {nms_outputs[1]};

    return {nms_outputs[0], nms_outputs[1]};
}

ONNX_OP("TRTBatchedNMS", OPSET_SINCE(1), ai_onnx::opset_1::trt_batched_nms, MMDEPLOY_DOMAIN);

}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
