// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_get_rows(const NodeContext& context) {
    num_inputs_check(context, 2, 2);

    int op_case = context.get_op_case();

    Output<Node> res;
    auto data = context.get_input(0);
    auto indices = context.get_input(1);

    // MoE gating-weight gather (op_case 10): data = probs [1,1,T,E], indices = selected experts
    // [1,1,T,K]; per-row (GatherElements) gather over the expert axis picks each token's K
    // selected-expert probs -> [1,1,T,K], distinct from the embedding-style row gather below.
    if (op_case == 10) {
        // Reshape to [1,T,K,1] for the broadcast-multiply with experts [1,T,K,n_embd]. Use an
        // explicit [1,-1,K,1] reshape (K is static, read via PartialShape to avoid .to_shape()
        // throwing when T is dynamic) instead of Squeeze+Unsqueeze, which the CPU plugin
        // implements as a Reshape internally and mis-infers the static pattern when T=1.
        const int64_t K = context.get_output_shape()[2].get_length();
        auto idx = std::make_shared<ov::op::v0::Convert>(indices, ov::element::i32);
        auto ge = std::make_shared<ov::op::v6::GatherElements>(data, idx, -1);  // [1,1,T,K]
        auto col = std::make_shared<ov::op::v1::Reshape>(
            ge,
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, -1, K, 1}),
            false);  // [1,T,K,1]
        return rename_outputs_with_suffix({col}, context.get_name());
    }

    if (op_case == 2) {
        // The input comes from a VIEW
        indices = process_view_input(context, 1);
    }

    // data[1,b,x,y] ind[1,1,b,x'] test-backend-ops case
    // data[x,y] ind[1,1,1,x'] normal case
    indices =
        std::make_shared<ov::op::v0::Squeeze>(indices, ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 1}));
    if (data.get_partial_shape().rank() == 4) {
        if (!(data.get_partial_shape()[1].is_dynamic()) && data.get_partial_shape()[1].get_length() == 1) {
            // Work-around for a bug in ov cpu plugin for test-backend-ops
            data = std::make_shared<ov::op::v0::Squeeze>(data,
                                                         ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 1}));
            auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
            res = std::make_shared<ov::op::v8::Gather>(data, indices, axis);
        } else {
            auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
            data =
                std::make_shared<ov::op::v0::Squeeze>(data, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
            res = std::make_shared<ov::op::v8::Gather>(data, indices, axis, 1);
        }
    } else {
        auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
        res = std::make_shared<ov::op::v8::Gather>(data, indices, axis);
    }

    if (res.get_element_type() != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
    }
    // The two Squeezes above dropped the leading axes; restore ggml's rank-4 form.
    res = std::make_shared<ov::op::v0::Unsqueeze>(res, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
