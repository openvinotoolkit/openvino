// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

void normalize_block_shape_pads_crops(const NodeContext& node,
                                      Output<Node>& block_shape,
                                      Output<Node>& pads_crops_begin,
                                      Output<Node>& pads_crops_end) {
    auto input = node.get_input(0);
    block_shape = node.get_input(1);
    auto paddings_crops = node.get_input(2);

    // make sure that paddings_crops and block_shape to have the same type
    paddings_crops = make_shared<Convert>(paddings_crops, block_shape.get_element_type())->output(0);

    // compute the input rank and a shape of block_shape [M]
    // it is needed to normalize block_shape, pads_begin and pads_end
    auto input_rank = compute_subgraph_scalar_rank(input, element::i32, false);
    auto M = make_shared<ShapeOf>(block_shape, element::i32);

    // compute a number of padding elements required to normalize block_shape
    // in the beggining and in the end
    auto num_begin = make_shared<Constant>(element::i32, Shape{1}, 1);
    auto num_end = make_shared<Subtract>(input_rank, M)->output(0);
    num_end = make_shared<Subtract>(num_end, num_begin)->output(0);

    // split paddings of shape [M, 2] into pads_begin and pads_end of shape [M]
    auto split_axis = make_shared<Constant>(element::i32, Shape{}, 1);
    auto paddings_crops_split = make_shared<Split>(paddings_crops, split_axis, 2);
    pads_crops_begin = make_shared<Squeeze>(paddings_crops_split->output(0), split_axis)->output(0);
    pads_crops_end = make_shared<Squeeze>(paddings_crops_split->output(1), split_axis)->output(0);

    // normalize block_shape to have it of the same length as the input rank
    auto one_const = make_shared<Constant>(block_shape.get_element_type(), Shape{}, 1);
    block_shape = make_shared<Pad>(block_shape, num_begin, num_end, one_const, ov::op::PadMode::CONSTANT)->output(0);

    // normalize pads_begin and pads_end to have it of the same length as the input rank
    auto zero_const = make_shared<Constant>(pads_crops_begin.get_element_type(), Shape{}, 0);
    auto pad_mode = ov::op::PadMode::CONSTANT;
    pads_crops_begin = make_shared<Pad>(pads_crops_begin, num_begin, num_end, zero_const, pad_mode)->output(0);
    pads_crops_end = make_shared<Pad>(pads_crops_end, num_begin, num_end, zero_const, pad_mode)->output(0);
}

OutputVector translate_space_to_batch_nd_op(const NodeContext& node) {
    default_op_checks(node, 3, {"SpaceToBatchND"});
    auto input = node.get_input(0);
    Output<Node> block_shape, pads_begin, pads_end;

    normalize_block_shape_pads_crops(node, block_shape, pads_begin, pads_end);
    auto space_to_batch = make_shared<SpaceToBatch>(input, block_shape, pads_begin, pads_end);
    set_node_name(node.get_name(), space_to_batch);
    return {space_to_batch};
}

OutputVector translate_batch_to_space_nd_op(const NodeContext& node) {
    default_op_checks(node, 3, {"BatchToSpaceND"});
    auto input = node.get_input(0);
    Output<Node> block_shape, crops_begin, crops_end;

    normalize_block_shape_pads_crops(node, block_shape, crops_begin, crops_end);
    auto batch_to_space = make_shared<BatchToSpace>(input, block_shape, crops_begin, crops_end);
    set_node_name(node.get_name(), batch_to_space);
    return {batch_to_space};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
