// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/canonicalize_moe_router.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::pass {

namespace {

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v8 = ov::op::v8;
namespace v12 = ov::op::v12;

// True if the input is a Constant filled with zeros, or a Broadcast of a scalar zero Constant.
bool is_zero_filled(const Output<Node>& output) {
    auto node = output.get_node_shared_ptr();
    if (auto bcast = ov::as_type_ptr<v3::Broadcast>(node)) {
        node = bcast->get_input_node_shared_ptr(0);
    }
    auto c = ov::as_type_ptr<v0::Constant>(node);
    if (!c) {
        return false;
    }
    return ov::op::util::constantIsEqualTo(c, 0.0F);
}

}  // namespace

ov::pass::CanonicalizeMoeRouter::CanonicalizeMoeRouter() {
    MATCHER_SCOPE(CanonicalizeMoeRouter);

    // BMM output of the down projection: [E, T, H], where T = num_tokens.
    auto down_out = pattern::any_input(pattern::rank_equals(3));
    // Reshape to [E, B, S, H] (or [E, B, -1, H] dynamically).
    auto end_reshape = pattern::wrap_type<v1::Reshape>({down_out, pattern::any_input()});

    // Densified routing weights: ScatterElementsUpdate(zeros[T,E], topk_idx, topk_val, axis=1).
    auto scatter_data = pattern::any_input();
    auto scatter_idx = pattern::any_input(pattern::rank_equals(2));
    auto scatter_upd = pattern::any_input(pattern::rank_equals(2));
    auto scatter_axis = pattern::wrap_type<v0::Constant>(pattern::value_matches("1"));
    auto scatter = pattern::wrap_type<v3::ScatterElementsUpdate, v12::ScatterElementsUpdate>(
        {scatter_data, scatter_idx, scatter_upd, scatter_axis});

    // Transpose [T,E] -> [E,T]; Reshape to [E,B,*]; Unsqueeze last dim to broadcast over H.
    auto transpose_perm = pattern::wrap_type<v0::Constant>(pattern::value_matches("1, 0"));
    auto routing_transpose = pattern::wrap_type<v1::Transpose>({scatter, transpose_perm});
    auto routing_reshape = pattern::wrap_type<v1::Reshape>({routing_transpose, pattern::any_input()});
    auto routing_unsq = pattern::wrap_type<v0::Unsqueeze>({routing_reshape, pattern::any_input()});

    auto multiply = pattern::wrap_type<v1::Multiply>({end_reshape, routing_unsq});
    auto reduce_axis = pattern::wrap_type<v0::Constant>(pattern::value_matches("0"));
    auto reduce_sum =
        pattern::wrap_type<v1::ReduceSum>({multiply, reduce_axis}, pattern::attrs_match({{"keep_dims", false}}));

    matcher_pass_callback callback = [=](pattern::Matcher& m) -> bool {
        const auto& pm = m.get_pattern_value_map();

        // Guardrail: scatter data must be all zeros — otherwise this is not a routing densification.
        if (!is_zero_filled(pm.at(scatter_data))) {
            return false;
        }

        const auto reduce_sum_node = pm.at(reduce_sum).get_node_shared_ptr();

        const auto down_out_value = pm.at(down_out);
        const auto topk_idx_value = pm.at(scatter_idx);
        const auto topk_val_value = pm.at(scatter_upd);

        // [E,T,H] -> [T,E,H]
        const auto perm_const = v0::Constant::create(element::i64, Shape{3}, {1, 0, 2});
        const auto transpose = std::make_shared<v1::Transpose>(down_out_value, perm_const);

        // [T,E,H] -> Gather(axis=1, batch_dims=1, topk_idx[T,K]) -> [T,K,H]
        const auto gather_axis_const = v0::Constant::create(element::i64, Shape{}, {1});
        const auto gather = std::make_shared<v8::Gather>(transpose,
                                                         topk_idx_value,
                                                         gather_axis_const,
                                                         /*batch_dims=*/1);

        // topk_val[T,K] -> Unsqueeze(-1) -> [T,K,1]
        const auto unsq_axis_const = v0::Constant::create(element::i64, Shape{1}, {-1});
        const auto unsq = std::make_shared<v0::Unsqueeze>(topk_val_value, unsq_axis_const);

        // [T,K,H] * [T,K,1] -> [T,K,H]; ReduceSum(axis=1, keep_dims=false) -> [T,H].
        const auto multiply_node = std::make_shared<v1::Multiply>(gather, unsq);
        const auto reduce_axis_const = v0::Constant::create(element::i64, Shape{1}, {1});
        const auto new_reduce = std::make_shared<v1::ReduceSum>(multiply_node, reduce_axis_const, false);

        // Old reduce produced [B,S,H] (or some [..,H]); new produces [T,H]. Both have the same total
        // element count, so the downstream Reshape to original shape adapts unchanged.
        ov::copy_runtime_info(reduce_sum_node,
                              {perm_const,
                               transpose,
                               gather_axis_const,
                               gather,
                               unsq_axis_const,
                               unsq,
                               multiply_node,
                               reduce_axis_const,
                               new_reduce});
        new_reduce->set_friendly_name(reduce_sum_node->get_friendly_name());
        ov::replace_node(reduce_sum_node, new_reduce);
        return true;
    };

    auto matcher = std::make_shared<pattern::Matcher>(reduce_sum, matcher_name);
    register_matcher(matcher, callback);
}

}  // namespace ov::pass
