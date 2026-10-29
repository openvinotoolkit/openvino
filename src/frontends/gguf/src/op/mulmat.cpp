// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <climits>
#include <cstdint>
#include <memory>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/op_types.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

OutputVector translate_mulmat(const NodeContext& context) {
    num_inputs_check(context, 2, 2);

    int op_case = context.get_attribute<int>("op_case", 0);

    ov::Output<Node> res;
    ov::Output<ov::Node> B = context.get_input(0);
    ov::Output<ov::Node> A = context.get_input(1);
    const auto activation = A;
    auto weight = B.get_node_shared_ptr();
    while (weight->get_input_size() > 0)
        weight = weight->get_input_node_shared_ptr(0);
    const bool u2_weights =
        ov::is_type<ov::op::v0::Constant>(weight) && weight->get_output_element_type(0) == ov::element::u2;
    const auto& shape = A.get_partial_shape();
    const bool rank3_output = context.get_attribute<int>("activation_rank", 4) == 3;
    const bool rank3_projection = u2_weights && shape.rank() == 4 && shape[1] == 1 && shape[3].is_static() &&
                                  B.get_partial_shape().rank() == 2 && op_case != 3;
    if (rank3_output) {
        FRONT_END_OP_CONVERSION_CHECK(shape.rank() == 4 && shape[1] == 1 && shape[3].is_static(),
                                      "Rank-3 projection requires [batch, 1, tokens, features] activations");
    }
    if (rank3_projection)
        A = context.get_rank3_activation(A);
    A = apply_hadamard(context, A, u2_weights);
    if (rank3_output && !rank3_projection)
        A = context.get_rank3_activation(A);

    const auto restore_projection = [&](const ov::Output<Node>& output) -> ov::Output<Node> {
        if (!rank3_projection || rank3_output)
            return output;
        const auto restored_shape = std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{get_dimensions(activation, {0, 1, 2}), get_dimensions(output, {-1})},
            0);
        return std::make_shared<ov::op::v1::Reshape>(output, restored_shape, false);
    };

    bool transpose_b = true;
    if (op_case == 2) {
        B = B.get_node_shared_ptr()->input_value(0);
        transpose_b = false;
    } else if (op_case == 3) {
        B = process_view_input(context, 0);
        A = process_view_input(context, 1);
    }
    // GGML MUL_MAT produces F32, including when both operands are F16.
    A = context.get_f32_activation(A);
    if (B.get_element_type() != ov::element::f32) {
        B = std::make_shared<ov::op::v0::Convert>(B, ov::element::f32);
    }

    auto B_shape = context.get_input_shape(0);
    auto A_shape = context.get_input_shape(1);
    if (A_shape[1].is_dynamic() || B_shape[1].is_dynamic()) {
        // Variable encoder grids use equal or singleton batch axes. MatMul validates
        // their broadcast compatibility at runtime; do not freeze a grid dimension.
        res = std::make_shared<ov::op::v0::MatMul>(A, B, false, transpose_b);
        res = restore_projection(res);
        return rename_outputs_with_suffix({res}, context.get_name());
    }
    int64_t A_batch = A_shape[1].get_length();
    int64_t B_batch = B_shape[1].get_length();

    auto A_batch_larger = A_batch > B_batch;
    auto batch_large = A_batch_larger ? A_batch : B_batch;
    auto batch_small = A_batch_larger ? B_batch : A_batch;

    Output<Node> Z = A_batch_larger ? B : A;
    int64_t factor = batch_large / batch_small;
    if (factor > 1 && batch_small > 1) {
        auto batch_large_node = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_large});
        auto batch_small_node = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_small});
        auto factor_node = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{factor});

        auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
        auto Z_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(Z, unsqueeze_axes);

        auto broadcast_shape = ov::op::v0::Constant::create(ov::element::i64,
                                                            {5},
                                                            {(int64_t)1, (int64_t)1, factor, (int64_t)1, (int64_t)1});
        auto new_Z_shape =
            ov::op::v0::Constant::create(ov::element::i64,
                                         {4},
                                         {(int64_t)0, batch_large, (int64_t)-1, (int64_t)A_shape[3].get_length()});

        auto Z_broadcasted = std::make_shared<ov::op::v3::Broadcast>(Z_unsqueezed,
                                                                     broadcast_shape,
                                                                     ov::op::BroadcastType::BIDIRECTIONAL);
        Z = std::make_shared<ov::op::v1::Reshape>(Z_broadcasted, new_Z_shape, true);
    }
    if (A_batch_larger) {
        B = Z;
    } else {
        A = Z;
    }

    res = std::make_shared<ov::op::v0::MatMul>(A, B, false, transpose_b);
    res = restore_projection(res);

    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
