// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_non_max_suppression_op(const NodeContext& node) {
    auto boxes = node.get_input(0);
    auto scores = node.get_input(1);
    auto max_output_size = node.get_input(2);
    auto iou_threshold = node.get_input(3);

    auto axis = make_shared<Constant>(element::i64, Shape{1}, 0);
    auto boxes_unsqueezed = make_shared<Unsqueeze>(boxes, axis);

    auto axis_scores = make_shared<Constant>(element::i64, Shape{2}, vector<int64_t>{0, 1});
    auto scores_unsqueezed = make_shared<Unsqueeze>(scores, axis_scores);

    const auto& op_type = node.get_op_type();
    if (op_type == "NonMaxSuppressionV5") {
        auto score_threshold = node.get_input(4);
        auto soft_nms_sigma = node.get_input(5);
        // todo: pad_to_max_output_size
        auto res = make_shared<NonMaxSuppression>(boxes_unsqueezed,
                                                  scores_unsqueezed,
                                                  max_output_size,
                                                  iou_threshold,
                                                  score_threshold,
                                                  soft_nms_sigma,
                                                  NonMaxSuppression::BoxEncodingType::CORNER,
                                                  false,
                                                  element::Type_t::i32);
        set_node_name(node.get_name(), res);
        return res->outputs();
    } else if (op_type == "NonMaxSuppressionV4") {
        auto score_threshold = node.get_input(4);
        // todo: pad_to_max_output_size
        auto res = make_shared<NonMaxSuppression>(boxes_unsqueezed,
                                                  scores_unsqueezed,
                                                  max_output_size,
                                                  iou_threshold,
                                                  score_threshold,
                                                  NonMaxSuppression::BoxEncodingType::CORNER,
                                                  false,
                                                  element::Type_t::i32);
        set_node_name(node.get_name(), res);
        return res->outputs();
    } else if (op_type == "NonMaxSuppressionV3") {
        auto score_threshold = node.get_input(4);
        auto res = make_shared<NonMaxSuppression>(boxes_unsqueezed,
                                                  scores_unsqueezed,
                                                  max_output_size,
                                                  iou_threshold,
                                                  score_threshold,
                                                  NonMaxSuppression::BoxEncodingType::CORNER,
                                                  false,
                                                  element::Type_t::i32);
        set_node_name(node.get_name(), res);
        return {res->output(0)};
    } else if (op_type == "NonMaxSuppressionV2" || op_type == "NonMaxSuppression") {
        auto res = make_shared<NonMaxSuppression>(boxes_unsqueezed,
                                                  scores_unsqueezed,
                                                  max_output_size,
                                                  iou_threshold,
                                                  NonMaxSuppression::BoxEncodingType::CORNER,
                                                  false,
                                                  element::Type_t::i32);
        set_node_name(node.get_name(), res);
        return {res->output(0)};
    }
    TENSORFLOW_OP_VALIDATION(node, false, "No translator found.");
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
