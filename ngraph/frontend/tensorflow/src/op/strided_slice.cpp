// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <default_opset.h>

#include <op_table.hpp>

using namespace std;
using namespace ngraph;
using namespace ngraph::frontend::tensorflow::detail;

#if 0

namespace tensorflow {
namespace ngraph_bridge {

OutputVector TranslateStridedSliceOp(
    const NodeContext& node) {
  Output<Node> ng_input;
  TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));

  int32_t begin_mask, end_mask, new_axis_mask, shrink_axis_mask, ellipsis_mask;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "begin_mask", &begin_mask));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "end_mask", &end_mask));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "new_axis_mask", &new_axis_mask));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op->attrs(), "shrink_axis_mask", &shrink_axis_mask));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "ellipsis_mask", &ellipsis_mask));

  NGRAPH_VLOG(5) << "strided slice attributes: "
                 << "  begin mask: " << begin_mask << "  end mask: " << end_mask
                 << "  new axis mask: " << new_axis_mask
                 << "  shrink axis mask: " << shrink_axis_mask
                 << "  ellipsis mask: " << ellipsis_mask;

  std::vector<int64_t> begin_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(ng_op_map, op, 1, static_input_map, &begin_vec));
  std::vector<int64_t> end_vec;
  TF_RETURN_IF_ERROR(GetStaticInputVector(ng_op_map, op, 2, static_input_map, &end_vec));
  std::vector<int64_t> stride_vec;
  TF_RETURN_IF_ERROR(
      GetStaticInputVector(ng_op_map, op, 3, static_input_map, &stride_vec));

  auto begin = ConstructNgNode<opset::Constant>(
      node.get_name(), element::i64, Shape{begin_vec.size()}, begin_vec);
  auto end = ConstructNgNode<opset::Constant>(
      node.get_name(), element::i64, Shape{end_vec.size()}, end_vec);
  auto strides = ConstructNgNode<opset::Constant>(
      node.get_name(), element::i64, Shape{stride_vec.size()}, stride_vec);

  auto mask_to_vec = [](int32_t mask) {
    auto length = sizeof(mask) * CHAR_BIT;
    std::vector<int64_t> vec(length, 0);
    if (mask == 0) {
      return vec;
    }
    for (auto i = 0; i < length; ++i) {
      if ((unsigned char)(mask >> i & 0x01) == 1) {
        vec[i] = 1;
      }
    }
    return vec;
  };

  SaveNgOp(
      ng_op_map, node.get_name(),
      ConstructNgNode<opset::StridedSlice>(
          node.get_name(), ng_input, begin, end, strides, mask_to_vec(begin_mask),
          mask_to_vec(end_mask), mask_to_vec(new_axis_mask),
          mask_to_vec(shrink_axis_mask), mask_to_vec(ellipsis_mask)));
  return Status::OK();
}

}
}
#endif