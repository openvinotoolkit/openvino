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
OutputVector translate_batch_nd_and_space_nd_op(const NodeContext& node) {
    auto input = node.get_input(0);
    auto block_shape = node.get_input(1);
    auto crops = node.get_input(2);

    // ng_crops should be of shape N=[ng_input.get_shape()).size()]
    // But TF's ng_crops input is limited only to the spatial dimensions (neither
    // batch nor innermost),
    // which would mean OV inputs have missing ng_crops[0] and ng_crops[N].
    // Hence, pad ng_crops with zeros at both ends

    // return with input if rank < 2 as OV's impl doesn't support it
    const auto& input_pshape = input.get_partial_shape();
    const auto& block_shape_pshape = block_shape.get_partial_shape();
    if (input_pshape.rank().is_static() && block_shape_pshape.rank().is_static()) {
        auto N = input_pshape.rank().get_length();
        if (N < 2)
            return {input};
    } else {
        // TODO: support dynamic rank
        TENSORFLOW_OP_VALIDATION(node, false, "Dynamic rank is not supported.");
    }

    auto N = input_pshape.rank().get_length();

    // TODO: support dynamic shape
    TENSORFLOW_OP_VALIDATION(node,
                             block_shape_pshape[0].is_static(),
                             "First dimension of block_shape input should be static.");
    auto M = static_cast<int64_t>(block_shape_pshape[0].get_length());

    auto padded_crops =
        make_shared<Pad>(crops,
                         make_shared<Constant>(crops.get_element_type(), Shape{2}, std::vector<int64_t>{1, 0}),
                         make_shared<Constant>(crops.get_element_type(), Shape{2}, std::vector<int64_t>{N - M - 1, 0}),
                         ov::op::PadMode::CONSTANT);

    // Padding needs to be done for block_shape as done for crops above but with
    // value=1
    auto padded_block_shape = make_shared<Pad>(
        block_shape,
        make_shared<Constant>(block_shape.get_element_type(), Shape{1}, std::vector<int64_t>{1}),
        make_shared<Constant>(block_shape.get_element_type(), Shape{1}, std::vector<int64_t>{N - M - 1}),
        make_shared<Constant>(block_shape.get_element_type(), Shape{}, 1),
        ov::op::PadMode::CONSTANT);

    auto target_axis = make_shared<Constant>(element::i64, Shape{}, 1);
    // split into two 1-D vectors crops_begin and crops_end along axis 1
    auto crops_split = make_shared<Split>(padded_crops, target_axis, 2);

    // crops: [[0, 1], [1, 2], ...]
    // crops_split: [[[0], [1]], [[1], [2]], ...]
    // crops_begin: [0, 1, ...], crops_end: [1, 2, ...]
    auto axes = make_shared<Constant>(element::i32, Shape{}, -1);
    auto crops_begin = make_shared<Squeeze>(crops_split->outputs()[0], axes);
    auto crops_end = make_shared<Squeeze>(crops_split->outputs()[1], axes);

    if (node.get_op_type() == "BatchToSpaceND") {
        auto res = make_shared<BatchToSpace>(input, padded_block_shape, crops_begin, crops_end);
        set_node_name(node.get_name(), res);
        return res->outputs();
    } else if (node.get_op_type() == "SpaceToBatchND") {
        auto res = make_shared<SpaceToBatch>(input, padded_block_shape, crops_begin, crops_end);
        set_node_name(node.get_name(), res);
        return res->outputs();
    }
    TENSORFLOW_OP_VALIDATION(node, false, "No translator found.");
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
