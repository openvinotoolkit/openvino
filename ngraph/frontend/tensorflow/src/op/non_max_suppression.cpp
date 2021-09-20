// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <default_opset.h>

#include <op_table.hpp>
#include <tensorflow_frontend/node_context.hpp>

using namespace std;
using namespace ngraph;
using namespace ngraph::frontend::tensorflow::detail;

#if 0

namespace tensorflow {
namespace ngraph_bridge {

OutputVector TranslateNonMaxSuppressionV2Op(
    const NodeContext& node) {
  Output<Node> ng_boxes, ng_scores, ng_unused, ng_iou_threshold;
  TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_boxes, ng_scores,
                                   ng_unused, ng_iou_threshold));

  auto ng_axis_boxes = ConstructNgNode<opset::Constant>(
      node.get_name(), element::i64, Shape{1}, std::vector<int64_t>({0}));
  auto ng_boxes_unsqueezed =
      ConstructNgNode<opset::Unsqueeze>(node.get_name(), ng_boxes, ng_axis_boxes);

  auto ng_axis_scores = ConstructNgNode<opset::Constant>(
      node.get_name(), element::i64, Shape{1}, std::vector<int64_t>({0}));
  auto ng_scores_unsqueezed1 =
      ConstructNgNode<opset::Unsqueeze>(node.get_name(), ng_scores, ng_axis_scores);
  auto ng_scores_unsqueezed2 = ConstructNgNode<opset::Unsqueeze>(
      node.get_name(), ng_scores_unsqueezed1, ng_axis_scores);

  std::vector<int> max_output_size;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(ng_op_map, op, 2, static_input_map, &max_output_size));

  // max_output_size must be scalar
  if (max_output_size.size() != 1) {
    return errors::InvalidArgument(
        "NonMaxSuppression Op: max_output_size of nms must be scalar " +
        to_string(max_output_size.size()));
  }

  auto ng_max_output_size = ConstructNgNode<opset::Constant>(
      node.get_name(), element::i64, Shape{}, max_output_size[0]);
  NGRAPH_VLOG(5) << "ng_max_output_size " << max_output_size[0];

  auto ng_nmsv = ConstructNgNode<opset::NonMaxSuppression>(
      node.get_name(), ng_boxes_unsqueezed, ng_scores_unsqueezed2,
      ng_max_output_size, ng_iou_threshold,
      opset::NonMaxSuppression::BoxEncodingType::CORNER, false,
      ngraph::element::Type_t::i32);

  auto begin = ConstructNgNode<opset::Constant>(
      node.get_name(), element::i64, Shape{2}, std::vector<int64_t>({0, 2}));
  auto end = ConstructNgNode<opset::Constant>(
      node.get_name(), element::i64, Shape{2},
      std::vector<int64_t>({max_output_size[0], 3}));
  auto ng_nmsv_slice = ConstructNgNode<opset::StridedSlice>(
      node.get_name(), ng_nmsv, begin, end, std::vector<int64_t>{0, 0},
      std::vector<int64_t>{0, 0}, std::vector<int64_t>{0, 0},
      std::vector<int64_t>{0, 1});

  Builder::SetTracingInfo(node.get_name(), ng_nmsv_slice);
  SaveNgOp(ng_op_map, node.get_name(), ng_nmsv_slice);
  return Status::OK();
}
}
}

#endif