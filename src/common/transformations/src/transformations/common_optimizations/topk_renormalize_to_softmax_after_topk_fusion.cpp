// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/topk_renormalize_to_softmax_after_topk_fusion.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/util/topk_base.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v8 = ov::op::v8;

namespace ov::pass {

TopkRenormalizeToSoftmaxAfterTopkFusion::TopkRenormalizeToSoftmaxAfterTopkFusion() {
    MATCHER_SCOPE(TopkRenormalizeToSoftmaxAfterTopkFusion);

    auto p_logits = pattern::any_input();
    // Softmax feeds only this TopK; else dropping it changes other consumers.
    auto p_softmax = pattern::wrap_type<v1::Softmax, v8::Softmax>({p_logits}, pattern::consumers_count(1));
    auto p_topk_k = pattern::any_input();
    // TopK.values (port 0) feeds only ReduceSum and Divide; extras would observe softmax-probs replaced by raw logits.
    auto p_topk =
        pattern::wrap_type<ov::op::util::TopKBase>({p_softmax, p_topk_k},
                                                   pattern::consumers_count(2) && pattern::output_index_matches(0));
    auto p_red_axes = pattern::wrap_type<v0::Constant>();
    auto p_reduce =
        pattern::wrap_type<v1::ReduceSum>({p_topk, p_red_axes}, pattern::attrs_match({{"keep_dims", true}}));
    auto p_divide = pattern::wrap_type<v1::Divide>({p_topk, p_reduce}, pattern::has_static_rank());

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pm = m.get_pattern_value_map();

        auto softmax = pm.at(p_softmax).get_node_shared_ptr();
        auto topk = pm.at(p_topk).get_node_shared_ptr();
        auto reduce = pm.at(p_reduce).get_node_shared_ptr();
        auto divide = pm.at(p_divide).get_node_shared_ptr();

        const auto rank = divide->get_output_partial_shape(0).rank();

        // Recover Softmax axis (normalized).
        int64_t softmax_axis = 0;
        if (auto sm8 = ov::as_type_ptr<v8::Softmax>(softmax)) {
            softmax_axis = static_cast<int64_t>(ov::util::try_normalize_axis(sm8->get_axis(), rank, *sm8));
        } else if (auto sm1 = ov::as_type_ptr<v1::Softmax>(softmax)) {
            softmax_axis = static_cast<int64_t>(sm1->get_axis());
        } else {
            return false;
        }

        // TopKBase unifies get_axis() across v1/v3/v11; get_axis() returns normalized axis.
        const auto topk_axis = static_cast<int64_t>(ov::as_type_ptr<ov::op::util::TopKBase>(topk)->get_axis());

        auto red_axes_const = ov::as_type_ptr<v0::Constant>(pm.at(p_red_axes).get_node_shared_ptr());
        if (!red_axes_const) {
            return false;
        }
        auto axes = red_axes_const->cast_vector<int64_t>();
        if (axes.size() != 1) {
            return false;
        }
        const auto reduce_axis = static_cast<int64_t>(ov::util::try_normalize_axis(axes[0], rank, *red_axes_const));
        if (reduce_axis != topk_axis || softmax_axis != topk_axis) {
            return false;
        }

        // Build a new TopK consuming the Softmax's input (logits) directly.
        // Indices are unchanged (Softmax is strictly monotonic); values become the raw logits' top-k.
        auto new_topk = topk->clone_with_new_inputs({pm.at(p_logits), pm.at(p_topk_k)});
        new_topk->set_friendly_name(topk->get_friendly_name());

        // Build a new SoftMax over new TopK.values
        std::shared_ptr<ov::Node> new_softmax;
        if (ov::as_type_ptr<v8::Softmax>(softmax)) {
            new_softmax = std::make_shared<v8::Softmax>(new_topk->output(0), softmax_axis);
        } else {
            new_softmax = std::make_shared<v1::Softmax>(new_topk->output(0), static_cast<size_t>(softmax_axis));
        }

        new_softmax->set_friendly_name(divide->get_friendly_name());

        copy_runtime_info({softmax, topk, reduce, divide}, {new_topk, new_softmax});

        // Swap TopK first so indices output (port 1) is fully transferred
        // Then replace Divide with the new Softmax
        // the old Softmax/Reduce/Divide become orphans and are removed by DCE.
        replace_node(topk, new_topk);
        replace_node(divide, new_softmax);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(p_divide, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::pass
