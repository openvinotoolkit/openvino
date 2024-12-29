// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/transpose.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_rot90(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    auto input = context.get_input(0);
    int k = context.input_is_none(1) ? 1 : context.const_input<int64_t>(1);
    std::vector<int64_t> dims = context.input_is_none(2) ? std::vector<int64_t>{0, 1}
                                                         : context.const_input<std::vector<int64_t>>(2);
    const auto& partial_shape = input.get_partial_shape();
    const auto ndims = partial_shape.rank().get_length();

    PYTORCH_OP_CONVERSION_CHECK(dims.size() == 2,
                                "Expected total rotation dims == 2, but got dims = ",
                                dims.size());
    PYTORCH_OP_CONVERSION_CHECK(ndims >= 2,
                                "Expected total dims >= 2, but got total dims = ",
                                ndims);
    PYTORCH_OP_CONVERSION_CHECK(dims[0] != dims[1],
                                "Rotation dimensions must be different, but got dim0 = " +
                                    std::to_string(dims[0]) + " and dim1 = " + std::to_string(dims[1]));

    for (auto& dim : dims) {
        dim = (dim + ndims) % ndims;
    }

    k = k % 4;
    Output<Node> rotated;

    if (k == 1 || k == 3) {
        int64_t flip_dim = (k == 1) ? dims[1] : dims[0];
        auto flip_dims = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {flip_dim}));
        auto flipped = create_flip(input, flip_dims);
        std::vector<int64_t> perm_values(ndims);
        std::iota(perm_values.begin(), perm_values.end(), 0);
        std::swap(perm_values[dims[0]], perm_values[dims[1]]);
        auto perm = context.mark_node(
            v0::Constant::create(element::i32, Shape{static_cast<size_t>(ndims)}, perm_values));
        rotated = context.mark_node(std::make_shared<v1::Transpose>(flipped, perm));
    } else if (k == 2) {
        size_t dims_size = dims.size();
        auto flip_dims = context.mark_node(v0::Constant::create(element::i32, Shape{dims_size}, dims));
        rotated = create_flip(input, flip_dims);
    } else {
        rotated = input;
    }

    return {rotated};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
