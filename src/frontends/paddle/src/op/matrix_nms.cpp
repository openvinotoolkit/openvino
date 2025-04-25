// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/frontend/paddle/visibility.hpp"
#include "openvino/opsets/opset8.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs matrix_nms(const NodeContext& node) {
    using namespace opset8;
    using namespace element;

    auto bboxes = node.get_input("BBoxes");
    auto scores = node.get_input("Scores");

    auto score_threshold = node.get_attribute<float>("score_threshold");
    auto post_threshold = node.get_attribute<float>("post_threshold");
    auto nms_top_k = node.get_attribute<int>("nms_top_k");
    auto keep_top_k = node.get_attribute<int>("keep_top_k");
    auto background_class = node.get_attribute<int>("background_label");

    auto gaussian_sigma = node.get_attribute<float>("gaussian_sigma");
    auto use_gaussian = node.get_attribute<bool>("use_gaussian");
    auto decay_function = MatrixNms::DecayFunction::LINEAR;
    if (use_gaussian) {
        decay_function = MatrixNms::DecayFunction::GAUSSIAN;
    }

    auto out_names = node.get_output_names();
    PADDLE_OP_CHECK(node,
                    out_names.size() == 3 || out_names.size() == 2,
                    "Unexpected number of outputs of MatrixNMS: " + std::to_string(out_names.size()));

    element::Type type_num = i32;
    bool return_rois_num = true;
    auto it = std::find(out_names.begin(), out_names.end(), "RoisNum");
    if (it != out_names.end()) {
        type_num = node.get_out_port_type("RoisNum");
    } else {
        return_rois_num = false;
    }

    auto type_index = node.get_out_port_type("Index");
    PADDLE_OP_CHECK(node,
                    (type_index == i32 || type_index == i64) && (type_num == i32 || type_num == i64),
                    "Unexpected data type of outputs of MatrixNMS");

    auto normalized = node.get_attribute<bool>("normalized");

    NamedOutputs named_outputs;
    std::vector<Output<Node>> nms_outputs;
    MatrixNms::Attributes attrs;
    attrs.nms_top_k = nms_top_k;
    attrs.post_threshold = post_threshold;
    attrs.score_threshold = score_threshold;
    attrs.sort_result_type = MatrixNms::SortResultType::SCORE;
    attrs.keep_top_k = keep_top_k;
    attrs.background_class = background_class;
    attrs.normalized = normalized;
    attrs.output_type = type_index;
    attrs.sort_result_across_batch = false;
    attrs.decay_function = decay_function;
    attrs.gaussian_sigma = gaussian_sigma;

    nms_outputs = std::make_shared<MatrixNms>(bboxes, scores, attrs)->outputs();

    named_outputs["Out"] = {nms_outputs[0]};
    named_outputs["Index"] = {nms_outputs[1]};
    if (return_rois_num) {
        named_outputs["RoisNum"] = {nms_outputs[2]};

        if (type_num != type_index) {
            // adapter
            auto node_convert = std::make_shared<Convert>(nms_outputs[2], type_num);
            named_outputs["RoisNum"] = {node_convert};
        }
    }

    return named_outputs;
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
