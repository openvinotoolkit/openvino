// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/frontend/paddle/visibility.hpp"
#include "openvino/opsets/opset9.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs multiclass_nms(const NodeContext& node) {
    using namespace opset9;
    using namespace element;

    auto bboxes = node.get_input("BBoxes");
    auto scores = node.get_input("Scores");

    auto score_threshold = node.get_attribute<float>("score_threshold");
    auto iou_threshold = node.get_attribute<float>("nms_threshold");
    auto nms_top_k = node.get_attribute<int>("nms_top_k");
    auto keep_top_k = node.get_attribute<int>("keep_top_k");
    auto background_class = node.get_attribute<int>("background_label");
    auto nms_eta = node.get_attribute<float>("nms_eta");

    auto out_names = node.get_output_names();
    PADDLE_OP_CHECK(node, out_names.size() == 3, "Unexpected number of outputs of MulticlassNMS");

    auto type_index = node.get_out_port_type("Index");
    auto type_num = node.get_out_port_type("NmsRoisNum");
    PADDLE_OP_CHECK(node,
                    (type_index == i32 || type_index == i64) && (type_num == i32 || type_num == i64),
                    "Unexpected data type of outputs of MulticlassNMS: " + std::to_string(out_names.size()));

    auto normalized = node.get_attribute<bool>("normalized");

    NamedOutputs named_outputs;
    std::vector<Output<Node>> nms_outputs;
    MulticlassNms::Attributes attrs;
    attrs.nms_top_k = nms_top_k;
    attrs.iou_threshold = iou_threshold;
    attrs.score_threshold = score_threshold;
    attrs.sort_result_type = MulticlassNms::SortResultType::CLASSID;
    attrs.keep_top_k = keep_top_k;
    attrs.background_class = background_class;
    attrs.nms_eta = nms_eta;
    attrs.normalized = normalized;
    attrs.output_type = type_index;
    attrs.sort_result_across_batch = false;

    if (node.has_input("RoisNum")) {
        const auto roisnum = node.get_input("RoisNum");

        // transpose scores and boxes first
        auto input_order_scores = Constant::create(element::i64, {2}, {1, 0});
        auto transposed_scores = std::make_shared<Transpose>(scores, input_order_scores);

        auto input_order_boxes = Constant::create(element::i64, {3}, {1, 0, 2});
        auto transposed_boxes = std::make_shared<Transpose>(bboxes, input_order_boxes);

        nms_outputs = std::make_shared<MulticlassNms>(transposed_boxes, transposed_scores, roisnum, attrs)->outputs();
    } else {
        nms_outputs = std::make_shared<MulticlassNms>(bboxes, scores, attrs)->outputs();
    }

    named_outputs["Out"] = {nms_outputs[0]};
    named_outputs["Index"] = {nms_outputs[1]};
    named_outputs["NmsRoisNum"] = {nms_outputs[2]};

    if (type_num != type_index) {
        // adapter
        auto node_convert = std::make_shared<Convert>(nms_outputs[2], type_num);
        named_outputs["NmsRoisNum"] = {node_convert};
    }

    return named_outputs;
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
