// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_rot90(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    auto input = context.get_input(0);
    int k = context.input_is_none(1) ? 1 : context.const_input<int32_t>(1);
    auto dims = context.input_is_none(2) 
    ? context.mark_node(v0::Constant::create(element::i32, Shape{2}, {0,1})) 
    : get_input_as_i32(context, 2);
    const auto& partial_shape = input.get_partial_shape();
    const auto ndims = partial_shape.rank().get_length();

    std::shared_ptr<ov::Node> rank = std::make_shared<ov::op::v0::Constant>(
        ov::element::i32, ov::Shape{}, std::vector<int32_t>{static_cast<int32_t>(ndims)});
    auto dims_norm = normalize_axis(context, dims, rank);
    auto dims_const = std::dynamic_pointer_cast<v0::Constant>(dims_norm.get_node_shared_ptr());
    auto dims_values = dims_const->cast_vector<int32_t>();

    auto start = v0::Constant::create(element::i32, {}, {0});
    auto step = v0::Constant::create(element::i32, {}, {1});
    auto range = std::make_shared<v4::Range>(start, rank, step, element::i32);

    auto axis_0 = v0::Constant::create(element::i32, Shape{}, {0});
    auto dim0_node = std::make_shared<v0::Unsqueeze>(
        v0::Constant::create(element::i32, {}, {dims_values[0]}), axis_0);
    auto dim1_node = std::make_shared<v0::Unsqueeze>(
        v0::Constant::create(element::i32, {}, {dims_values[1]}), axis_0);

    auto indices = std::make_shared<v0::Concat>(OutputVector{dim0_node, dim1_node}, 0);
    auto updates = std::make_shared<v0::Concat>(
        OutputVector{dim1_node, dim0_node}, 0); 

    Output<Node> scatter = std::make_shared<v3::ScatterElementsUpdate>(
        range, indices, updates, axis_0);
    if (const auto scatter_const = ov::util::get_constant_from_source(scatter)) {
        scatter = context.mark_node(scatter_const);
    } else {
        context.mark_nodes(
            {start, step, range, axis_0, dim0_node, dim1_node, indices, updates, scatter.get_node_shared_ptr()});
    }
  
    PYTORCH_OP_CONVERSION_CHECK(dims_values.size() == 2,
                                "Expected total rotation dims == 2, but got dims = ",
                                dims_values.size());
    PYTORCH_OP_CONVERSION_CHECK(ndims >= 2,
                                "Expected total dims >= 2, but got total dims = ",
                                ndims);
    PYTORCH_OP_CONVERSION_CHECK(dims_values[0] != dims_values[1],
                                "Rotation dimensions must be different, but got dim0 = " +
                                    std::to_string(dims_values[0]) + " and dim1 = " + std::to_string(dims_values[1]));

    k = k % 4;
    Output<Node> rotated;

    if (k == 1 || k == 3) {
        Output<Node> flip_dims = (k ==1) ? dim1_node : dim0_node;
        auto flipped =  create_flip(input, flip_dims);
        rotated = context.mark_node(std::make_shared<v1::Transpose>(flipped, scatter));
    } else if (k == 2) {
        rotated = create_flip(input, dims_norm);
    } else {
        rotated = input;
    }

    return {rotated};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
