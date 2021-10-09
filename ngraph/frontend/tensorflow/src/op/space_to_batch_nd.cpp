// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ngraph::opset8;

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {
OutputVector TranslateBatchNDAndSpaceNDOp(const NodeContext& node) {
    auto input = node.get_ng_input(0);
    auto block_shape = node.get_ng_input(1);
    auto crops = node.get_ng_input(2);

    // ng_crops should be of shape N=[ng_input.get_shape()).size()]
    // But TF's ng_crops input is limited only to the spatial dimensions (neither
    // batch nor innermost),
    // which would mean ngraph inputs have missing ng_crops[0] and ng_crops[N].
    // Hence, pad ng_crops with zeros at both ends

    // return with input if rank < 2 as ngraph's impl doesn't support it
    const auto& input_pshape = input.get_partial_shape();
    const auto& block_shape_pshape = block_shape.get_partial_shape();
    if (input_pshape.rank().is_static() && block_shape_pshape.rank().is_static()) {
        auto N = input_pshape.rank().get_length();
        if (N < 2)
            return {input};
    } else {
        // todo (itikhono): support dynamic rank
        TF_OP_VALIDATION_CHECK(node, false, "Dynamic rank is not supported.");
    }

    auto N = input_pshape.rank().get_length();
    auto M = block_shape_pshape.rank().get_length();
    auto padded_crops =
        make_shared<Pad>(crops,
                         make_shared<Constant>(crops.get_element_type(), Shape{2}, std::vector<int64_t>{1, 0}),
                         make_shared<Constant>(crops.get_element_type(), Shape{2}, std::vector<int64_t>{N - M - 1, 0}),
                         ngraph::op::PadMode::CONSTANT);

    // Padding needs to be done for block_shape as done for crops above but with
    // value=1
    auto padded_block_shape = make_shared<Pad>(
        block_shape,
        make_shared<Constant>(block_shape.get_element_type(), Shape{1}, std::vector<int64_t>{1}),
        make_shared<Constant>(block_shape.get_element_type(), Shape{1}, std::vector<int64_t>{N - M - 1}),
        make_shared<Constant>(block_shape.get_element_type(), Shape{}, 1),
        ngraph::op::PadMode::CONSTANT);

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
        auto batch_to_space_nd = make_shared<BatchToSpace>(input, padded_block_shape, crops_begin, crops_end);
        return batch_to_space_nd->outputs();
    } else if (node.get_op_type() == "SpaceToBatchND") {
        auto space_to_batch_nd = make_shared<SpaceToBatch>(input, padded_block_shape, crops_begin, crops_end);
        return space_to_batch_nd->outputs();
    }
    TF_OP_VALIDATION_CHECK(node, false, "No translator found.");
}

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
