// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "input_model.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "ov_tensorflow/xla_data.pb.h"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

vector<int64_t> compute_non_contracting_dims(const NodeContext& node,
                                             const vector<int64_t>& batch_dims,
                                             const vector<int64_t>& contracting_dims,
                                             const Output<Node>& operand) {
    // combine two vectors of batch_dims and contracting_dims
    set<int64_t> unique_dims(batch_dims.begin(), batch_dims.end());
    unique_dims.insert(contracting_dims.begin(), contracting_dims.end());
    vector<int64_t> all_dims(unique_dims.begin(), unique_dims.end());

    TENSORFLOW_OP_VALIDATION(node,
                             operand.get_partial_shape().rank().is_static(),
                             "[TensorFlow Frontend] internal operation: XlaDotV2 expects inputs of static rank");

    int64_t operand_rank = operand.get_partial_shape().rank().get_length();
    vector<int64_t> non_contracting_dims;
    for (int64_t ind = 0; ind < operand_rank; ++ind) {
        if (find(all_dims.begin(), all_dims.end(), ind) == all_dims.end()) {
            non_contracting_dims.push_back(ind);
        }
    }

    return non_contracting_dims;
}

void insert_aux_dim(const NodeContext& node, Output<Node>& operand, vector<int64_t>& dims) {
    TENSORFLOW_OP_VALIDATION(node,
                             operand.get_partial_shape().rank().is_static(),
                             "[TensorFlow Frontend] internal operation: XlaDotV2 expects inputs of static rank");
    if (dims.size() == 0) {
        int64_t operand_rank = operand.get_partial_shape().rank().get_length();
        dims.push_back(operand_rank);
        auto unsqueeze_axis = make_shared<v0::Constant>(element::i64, Shape{1}, operand_rank);
        operand = make_shared<v0::Unsqueeze>(operand, unsqueeze_axis);
    }
}

void insert_aux_dims(const NodeContext& node,
                     Output<Node>& operand,
                     vector<int64_t>& batch_dims,
                     vector<int64_t>& contracting_dims,
                     vector<int64_t>& non_contract_dims) {
    insert_aux_dim(node, operand, batch_dims);
    insert_aux_dim(node, operand, contracting_dims);
    insert_aux_dim(node, operand, non_contract_dims);
}

Output<Node> compute_dims_shape(const Output<Node>& hs_shape, const vector<int64_t>& dims) {
    auto const_dims = make_shared<v0::Constant>(element::i64, Shape{dims.size()}, dims);
    auto gather_axis = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    auto dims_shape = make_shared<v8::Gather>(hs_shape, const_dims, gather_axis);
    return dims_shape;
}

Output<Node> compute_dims_size(const Output<Node>& hs_shape, const Output<Node>& dims) {
    auto gather_axis = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    auto dims_shape = make_shared<v8::Gather>(hs_shape, dims, gather_axis);
    auto dims_size = make_shared<v1::ReduceProd>(dims_shape, gather_axis, true);
    return {dims_size};
}

OutputVector translate_xla_dot_op(const NodeContext& node) {
    // see specification of XlaDotV2 here: https://www.tensorflow.org/xla/operation_semantics#dot
    default_op_checks(node, 2, {"XlaDotV2"});
    auto lhs = node.get_input(0);
    auto rhs = node.get_input(1);
    auto node_name = node.get_name();
    auto dimension_numbers_message = node.get_attribute<string>("dimension_numbers");
    ::xla::DotDimensionNumbers dimension_numbers;
    TENSORFLOW_OP_VALIDATION(
        node,
        dimension_numbers.ParseFromArray(dimension_numbers_message.data(),
                                         static_cast<int>(dimension_numbers_message.size())),
        "[TensorFlow Frontend] Incorrect input model: incorrect DotDimensionNumbers field for XlaDotV2 " + node_name);

    vector<int64_t> lhs_batch_dims(dimension_numbers.lhs_batch_dimensions().begin(),
                                   dimension_numbers.lhs_batch_dimensions().end());
    vector<int64_t> rhs_batch_dims(dimension_numbers.rhs_batch_dimensions().begin(),
                                   dimension_numbers.rhs_batch_dimensions().end());
    vector<int64_t> rhs_contract_dims(dimension_numbers.rhs_contracting_dimensions().begin(),
                                      dimension_numbers.rhs_contracting_dimensions().end());
    vector<int64_t> lhs_contract_dims(dimension_numbers.lhs_contracting_dimensions().begin(),
                                      dimension_numbers.lhs_contracting_dimensions().end());

    // compute non-contracting dimensions
    auto lhs_non_contract_dims = compute_non_contracting_dims(node, lhs_batch_dims, lhs_contract_dims, lhs);
    auto rhs_non_contract_dims = compute_non_contracting_dims(node, rhs_batch_dims, rhs_contract_dims, rhs);

    // compute the resulted shape before possible modification
    auto resulted_shape = make_shared<v0::Constant>(element::i64, Shape{0}, vector<int64_t>{})->output(0);
    bool apply_reshape = false;
    auto lhs_shape = make_shared<v3::ShapeOf>(lhs, element::i64);
    auto rhs_shape = make_shared<v3::ShapeOf>(rhs, element::i64);
    if (lhs_batch_dims.size() > 0) {
        auto batch_dims_shape = compute_dims_shape(lhs_shape, lhs_batch_dims);
        resulted_shape = make_shared<v0::Concat>(OutputVector{resulted_shape, batch_dims_shape}, 0);
        apply_reshape = true;
    }
    if (lhs_non_contract_dims.size() > 0) {
        auto lhs_non_contract_shape = compute_dims_shape(lhs_shape, lhs_non_contract_dims);
        resulted_shape = make_shared<v0::Concat>(OutputVector{resulted_shape, lhs_non_contract_shape}, 0);
        apply_reshape = true;
    }
    if (rhs_non_contract_dims.size() > 0) {
        auto rhs_non_contract_shape = compute_dims_shape(rhs_shape, rhs_non_contract_dims);
        resulted_shape = make_shared<v0::Concat>(OutputVector{resulted_shape, rhs_non_contract_shape}, 0);
        apply_reshape = true;
    }

    // take care of that at least one dimension of each type (batch, contracting, and non-contracting) exists
    // if it does not, insert it to the end
    insert_aux_dims(node, lhs, lhs_batch_dims, lhs_contract_dims, lhs_non_contract_dims);
    insert_aux_dims(node, rhs, rhs_batch_dims, rhs_contract_dims, rhs_non_contract_dims);

    // compute non-batch and non-contracting dimensions
    auto const_lhs_batch_dims = make_shared<v0::Constant>(element::i64, Shape{lhs_batch_dims.size()}, lhs_batch_dims);
    auto const_rhs_batch_dims = make_shared<v0::Constant>(element::i64, Shape{rhs_batch_dims.size()}, rhs_batch_dims);
    auto const_lhs_contract_dims =
        make_shared<v0::Constant>(element::i64, Shape{lhs_contract_dims.size()}, lhs_contract_dims);
    auto const_rhs_contract_dims =
        make_shared<v0::Constant>(element::i64, Shape{rhs_contract_dims.size()}, rhs_contract_dims);
    auto const_lhs_non_contract_dims =
        make_shared<v0::Constant>(element::i64, Shape{lhs_non_contract_dims.size()}, lhs_non_contract_dims);
    auto const_rhs_non_contract_dims =
        make_shared<v0::Constant>(element::i64, Shape{rhs_non_contract_dims.size()}, rhs_non_contract_dims);

    lhs_shape = make_shared<v3::ShapeOf>(lhs, element::i64);
    rhs_shape = make_shared<v3::ShapeOf>(rhs, element::i64);

    // compute a part of the input shape covering batch dimensions and non-contracting dimensions
    auto gather_axis = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    auto batch_dims_shape = compute_dims_shape(lhs_shape, lhs_batch_dims);

    // transpose both operand in a way to have dimensions in the order
    // [batch dims, non-contracting dims, contracting dims]
    auto lhs_transpose_order = make_shared<v0::Concat>(
        OutputVector{const_lhs_batch_dims, const_lhs_non_contract_dims, const_lhs_contract_dims},
        0);
    auto rhs_transpose_order = make_shared<v0::Concat>(
        OutputVector{const_rhs_batch_dims, const_rhs_non_contract_dims, const_rhs_contract_dims},
        0);
    lhs = make_shared<v1::Transpose>(lhs, lhs_transpose_order);
    rhs = make_shared<v1::Transpose>(rhs, rhs_transpose_order);

    // compute size of contracting dims and non-contracting dims for each operand
    auto lhs_contract_size = compute_dims_size(lhs_shape, const_lhs_contract_dims);
    auto rhs_contract_size = compute_dims_size(rhs_shape, const_rhs_contract_dims);
    auto lhs_non_contract_size = compute_dims_size(lhs_shape, const_lhs_non_contract_dims);
    auto rhs_non_contract_size = compute_dims_size(rhs_shape, const_rhs_non_contract_dims);

    // merge contracting and non-contracting dimensions to have operand
    // of a shape [batch dims, non-contracting dim size, contracting dims size]
    auto new_lhs_shape =
        make_shared<v0::Concat>(OutputVector{batch_dims_shape, lhs_non_contract_size, lhs_contract_size}, 0);
    auto new_rhs_shape =
        make_shared<v0::Concat>(OutputVector{batch_dims_shape, rhs_non_contract_size, rhs_contract_size}, 0);
    lhs = make_shared<v1::Reshape>(lhs, new_lhs_shape, false);
    rhs = make_shared<v1::Reshape>(rhs, new_rhs_shape, false);

    // execute MatMul that support batch matrix-multiplication
    // note that the second operand is transposed
    auto matmul = make_shared<v0::MatMul>(lhs, rhs, false, true)->output(0);
    if (apply_reshape) {
        matmul = make_shared<v1::Reshape>(matmul, resulted_shape, false);
    }

    set_node_name(node_name, matmul.get_node_shared_ptr());
    return {matmul};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
