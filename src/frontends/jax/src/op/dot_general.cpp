// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/frontend/jax/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace jax {
namespace op {

using namespace ov::op;

std::vector<int64_t> compute_non_contracting_dims(const NodeContext& node,
                                                  const std::vector<int64_t>& batch_dims,
                                                  const std::vector<int64_t>& contracting_dims,
                                                  const Output<Node>& operand) {
    // combine two vectors of batch_dims and contracting_dims
    std::set<int64_t> unique_dims(batch_dims.begin(), batch_dims.end());
    unique_dims.insert(contracting_dims.begin(), contracting_dims.end());
    std::vector<int64_t> all_dims(unique_dims.begin(), unique_dims.end());

    int64_t operand_rank = operand.get_partial_shape().rank().get_length();
    std::vector<int64_t> non_contracting_dims;
    for (int64_t ind = 0; ind < operand_rank; ++ind) {
        if (find(all_dims.begin(), all_dims.end(), ind) == all_dims.end()) {
            non_contracting_dims.push_back(ind);
        }
    }

    return non_contracting_dims;
}

void insert_aux_dim(const NodeContext& node, Output<Node>& operand, std::vector<int64_t>& dims) {
    if (dims.size() == 0) {
        int64_t operand_rank = operand.get_partial_shape().rank().get_length();
        dims.push_back(operand_rank);
        auto unsqueeze_axis = std::make_shared<v0::Constant>(element::i64, Shape{1}, operand_rank);
        operand = std::make_shared<v0::Unsqueeze>(operand, unsqueeze_axis);
    }
}

void insert_aux_dims(const NodeContext& node,
                     Output<Node>& operand,
                     std::vector<int64_t>& batch_dims,
                     std::vector<int64_t>& contracting_dims,
                     std::vector<int64_t>& non_contract_dims) {
    insert_aux_dim(node, operand, batch_dims);
    insert_aux_dim(node, operand, contracting_dims);
    insert_aux_dim(node, operand, non_contract_dims);
}

Output<Node> compute_dims_shape(const Output<Node>& hs_shape, const std::vector<int64_t>& dims) {
    auto const_dims = std::make_shared<v0::Constant>(element::i64, Shape{dims.size()}, dims);
    auto gather_axis = std::make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    auto dims_shape = std::make_shared<v8::Gather>(hs_shape, const_dims, gather_axis);
    return dims_shape;
}

Output<Node> compute_dims_size(const Output<Node>& hs_shape, const Output<Node>& dims) {
    auto gather_axis = std::make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    auto dims_shape = std::make_shared<v8::Gather>(hs_shape, dims, gather_axis);
    auto dims_size = std::make_shared<v1::ReduceProd>(dims_shape, gather_axis, true);
    return {dims_size};
}

OutputVector translate_dot_general(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    Output<Node> lhs = context.get_input(0);
    Output<Node> rhs = context.get_input(1);

    auto contract_dimensions = context.const_named_param<std::vector<std::vector<int64_t>>>("contract_dimensions");
    JAX_OP_CONVERSION_CHECK(contract_dimensions.size() == 2, "Contract_dimensions of dot_general must have size 2");
    auto lhs_contract_dims = contract_dimensions[0];
    auto rhs_contract_dims = contract_dimensions[1];

    std::vector<int64_t> lhs_batch_dims, rhs_batch_dims;
    if (context.has_param("batch_dimensions")) {
        auto batch_dimensions = context.const_named_param<std::vector<std::vector<int64_t>>>("batch_dimensions");
        JAX_OP_CONVERSION_CHECK(batch_dimensions.size() == 2, "Batch_dimensions of dot_general must have size 2");
        lhs_batch_dims = batch_dimensions[0];
        rhs_batch_dims = batch_dimensions[1];
    }

    // compute non-contracting dimensions
    auto lhs_non_contract_dims = compute_non_contracting_dims(context, lhs_batch_dims, lhs_contract_dims, lhs);
    auto rhs_non_contract_dims = compute_non_contracting_dims(context, rhs_batch_dims, rhs_contract_dims, rhs);

    // compute the resulted shape before possible modification
    auto resulted_shape = std::make_shared<v0::Constant>(element::i64, Shape{0}, std::vector<int64_t>{})->output(0);
    bool apply_reshape = false;
    auto lhs_shape = std::make_shared<v3::ShapeOf>(lhs, element::i64);
    auto rhs_shape = std::make_shared<v3::ShapeOf>(rhs, element::i64);
    if (lhs_batch_dims.size() > 0) {
        auto batch_dims_shape = compute_dims_shape(lhs_shape, lhs_batch_dims);
        resulted_shape = std::make_shared<v0::Concat>(OutputVector{resulted_shape, batch_dims_shape}, 0);
        apply_reshape = true;
    }
    if (lhs_non_contract_dims.size() > 0) {
        auto lhs_non_contract_shape = compute_dims_shape(lhs_shape, lhs_non_contract_dims);
        resulted_shape = std::make_shared<v0::Concat>(OutputVector{resulted_shape, lhs_non_contract_shape}, 0);
        apply_reshape = true;
    }
    if (rhs_non_contract_dims.size() > 0) {
        auto rhs_non_contract_shape = compute_dims_shape(rhs_shape, rhs_non_contract_dims);
        resulted_shape = std::make_shared<v0::Concat>(OutputVector{resulted_shape, rhs_non_contract_shape}, 0);
        apply_reshape = true;
    }

    // take care of that at least one dimension of each type (batch, contracting, and non-contracting) exists
    // if it does not, insert it to the end
    insert_aux_dims(context, lhs, lhs_batch_dims, lhs_contract_dims, lhs_non_contract_dims);
    insert_aux_dims(context, rhs, rhs_batch_dims, rhs_contract_dims, rhs_non_contract_dims);

    // compute non-batch and non-contracting dimensions
    auto const_lhs_batch_dims =
        std::make_shared<v0::Constant>(element::i64, Shape{lhs_batch_dims.size()}, lhs_batch_dims);
    auto const_rhs_batch_dims =
        std::make_shared<v0::Constant>(element::i64, Shape{rhs_batch_dims.size()}, rhs_batch_dims);
    auto const_lhs_contract_dims =
        std::make_shared<v0::Constant>(element::i64, Shape{lhs_contract_dims.size()}, lhs_contract_dims);
    auto const_rhs_contract_dims =
        std::make_shared<v0::Constant>(element::i64, Shape{rhs_contract_dims.size()}, rhs_contract_dims);
    auto const_lhs_non_contract_dims =
        std::make_shared<v0::Constant>(element::i64, Shape{lhs_non_contract_dims.size()}, lhs_non_contract_dims);
    auto const_rhs_non_contract_dims =
        std::make_shared<v0::Constant>(element::i64, Shape{rhs_non_contract_dims.size()}, rhs_non_contract_dims);

    lhs_shape = std::make_shared<v3::ShapeOf>(lhs, element::i64);
    rhs_shape = std::make_shared<v3::ShapeOf>(rhs, element::i64);

    // compute a part of the input shape covering batch dimensions and non-contracting dimensions
    auto gather_axis = std::make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    auto batch_dims_shape = compute_dims_shape(lhs_shape, lhs_batch_dims);

    // transpose both operand in a way to have dimensions in the order
    // [batch dims, non-contracting dims, contracting dims]
    auto lhs_transpose_order = std::make_shared<v0::Concat>(
        OutputVector{const_lhs_batch_dims, const_lhs_non_contract_dims, const_lhs_contract_dims},
        0);
    auto rhs_transpose_order = std::make_shared<v0::Concat>(
        OutputVector{const_rhs_batch_dims, const_rhs_non_contract_dims, const_rhs_contract_dims},
        0);
    lhs = std::make_shared<v1::Transpose>(lhs, lhs_transpose_order);
    rhs = std::make_shared<v1::Transpose>(rhs, rhs_transpose_order);

    // compute size of contracting dims and non-contracting dims for each operand
    auto lhs_contract_size = compute_dims_size(lhs_shape, const_lhs_contract_dims);
    auto rhs_contract_size = compute_dims_size(rhs_shape, const_rhs_contract_dims);
    auto lhs_non_contract_size = compute_dims_size(lhs_shape, const_lhs_non_contract_dims);
    auto rhs_non_contract_size = compute_dims_size(rhs_shape, const_rhs_non_contract_dims);

    // merge contracting and non-contracting dimensions to have operand
    // of a shape [batch dims, non-contracting dim size, contracting dims size]
    auto new_lhs_shape =
        std::make_shared<v0::Concat>(OutputVector{batch_dims_shape, lhs_non_contract_size, lhs_contract_size}, 0);
    auto new_rhs_shape =
        std::make_shared<v0::Concat>(OutputVector{batch_dims_shape, rhs_non_contract_size, rhs_contract_size}, 0);
    lhs = std::make_shared<v1::Reshape>(lhs, new_lhs_shape, false);
    rhs = std::make_shared<v1::Reshape>(rhs, new_rhs_shape, false);

    // execute MatMul that support batch matrix-multiplication
    // note that the second operand is transposed
    auto matmul = std::make_shared<v0::MatMul>(lhs, rhs, false, true)->output(0);
    if (apply_reshape) {
        matmul = std::make_shared<v1::Reshape>(matmul, resulted_shape, false);
    }

    return {matmul};
};

}  // namespace op
}  // namespace jax
}  // namespace frontend
}  // namespace ov