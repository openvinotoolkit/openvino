// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_moe_expert_sort.hpp"

#include <algorithm>
#include <memory>
#include <unordered_set>

#include "intel_gpu/op/moe_expert_sort.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/symbol.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {
using namespace ov::pass::pattern;
using namespace ov::op;
using ov::pass::operator|;

namespace {

// Ascending argsort of the 1D `data`: TopK with k = number of elements of data
std::shared_ptr<Node> argsort_pattern(const Output<Node>& data) {
    auto data_cvt = optional<v0::Convert>({data});
    auto shape_of = wrap_type<v0::ShapeOf, v3::ShapeOf>({data_cvt});
    auto k = wrap_type<v0::Squeeze, v15::Squeeze>({shape_of, any_input()}) | wrap_type<v0::Squeeze, v15::Squeeze>({shape_of});
    return wrap_type_strict_index<v1::TopK, v3::TopK, v11::TopK>({data, k}, {{"mode", "min"}, {"sort", "value"}});
}

}  // namespace

FuseMoEExpertSort::FuseMoEExpertSort(size_t max_num_experts) {
    auto topk_ids_m = any_input(rank_equals(2) && has_static_dim(1));
    auto flat_ids_m = wrap_type<v1::Reshape>({topk_ids_m, any_input()}, rank_equals(1));
    auto argsort_m = argsort_pattern(flat_ids_m);
    auto perm_m = optional<v0::Convert>({argsort_m->output(1)});
    auto sorted_ids_m = wrap_type<v7::Gather, v8::Gather>({flat_ids_m, perm_m, 0}, {{"batch_dims", 0}});
    auto sorted_ids_cvt_m = optional<v0::Convert>({sorted_ids_m});
    auto zeros_m = any_input(
        rank_equals(1) && has_static_shape() &&
        // moe_mask_gen runs a single work-group with a work-item per expert
        [max_num_experts](const Output<Node>& output) {
            return output.get_shape()[0] <= max_num_experts;
        } &&
        [](const Output<Node>& output) {
            return ov::op::util::is_constant_and_all_values_equal_int(output, 0);
        });
    auto ones_m = wrap_type<v1::Broadcast, v3::Broadcast>({wrap_type<v0::Constant>(value_matches("1")), any_input()});
    auto updates_m = optional<v8::Slice>({ones_m, any_input(), any_input(), any_input(), any_input()});
    auto counts_m = wrap_type<v12::ScatterElementsUpdate>({zeros_m, sorted_ids_cvt_m, updates_m, 0}, {{"reduction", "sum"}, {"use_init_val", true}});
    auto offsets_m = wrap_type<v0::CumSum>({counts_m, 0}, {{"exclusive", false}, {"reverse", false}});

    // Extra parts of the pattern. They only consume the permutation, so they are not reachable from the root (offsets) and are
    // optional: matched separately in the callback, on the consumers of the permutation.
    auto extra_range_m = wrap_type<v4::Range>({0, any_input(), 1});
    auto extra_unsqueeze_m = wrap_type<v0::Unsqueeze>({extra_range_m, any_input()}, shape_matches("?, 1"));
    auto extra_expand_m = wrap_type<v1::Broadcast, v3::Broadcast>({extra_unsqueeze_m, any_input()}, rank_equals(2) && has_static_dim(1));
    auto extra_token_src_m = wrap_type<v1::Reshape>({extra_expand_m, any_input()}, rank_equals(1));
    auto extra_token_m = wrap_type<v7::Gather, v8::Gather>({extra_token_src_m, perm_m, 0}, {{"batch_dims", 0}});
    // Inverse permutation: argsort of the permutation
    auto extra_inv_argsort_m = argsort_pattern(argsort_m->output(1));

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        const auto& main_pm = m.get_pattern_value_map();
        const auto perm_out = m.get_pattern_map().at(argsort_m)->output(1);
        const auto perm = main_pm.count(perm_m) ? main_pm.at(perm_m) : perm_out;

        // The extra parts are matched on top of the main map, so that they reuse its nodes
        PatternValueMap pm = main_pm;

        // Matches `matcher` on a consumer of `source`, on top of pm
        auto match_consumer = [&pm](Matcher& matcher, const Output<Node>& source) {
            for (const auto& input : source.get_target_inputs()) {
                for (const auto& candidate : input.get_node()->outputs()) {
                    if (matcher.match(candidate, pm)) {
                        return true;
                    }
                }
            }
            return false;
        };
        Matcher extra_token_matcher(extra_token_m);
        if (match_consumer(extra_token_matcher, perm)) {
            pm = extra_token_matcher.get_pattern_value_map();
        }
        Matcher extra_inv_matcher(extra_inv_argsort_m);
        if (match_consumer(extra_inv_matcher, perm_out)) {
            pm = extra_inv_matcher.get_pattern_value_map();
        }
        const bool has_token = pm.count(extra_token_m) != 0U;
        const bool has_inv = pm.count(extra_inv_argsort_m) != 0U;

        const auto topk_ids = pm.at(topk_ids_m);
        const auto offsets = pm.at(offsets_m);
        const size_t top_k = static_cast<size_t>(topk_ids.get_partial_shape()[1].get_length());
        const size_t num_experts = pm.at(zeros_m).get_shape()[0];

        Output<Node> token;
        if (has_token) {
            const auto& expand_shape = pm.at(extra_expand_m).get_partial_shape();
            const auto& topk_ids_shape = topk_ids.get_partial_shape();
            const bool same_top_k = static_cast<size_t>(expand_shape[1].get_length()) == top_k;
            // MoEExpertSort computes the token index from topk_ids, so the token source must enumerate the same T tokens.
            // A dynamic T can only be compared by symbol, set by SymbolicPropagation
            const bool same_num_tokens = (expand_shape[0].is_static() && expand_shape[0] == topk_ids_shape[0]) ||
                                         ov::symbol::are_equal(expand_shape[0].get_symbol(), topk_ids_shape[0].get_symbol());
            if (!same_top_k || !same_num_tokens) {
                return false;
            }
            token = pm.at(extra_token_m);
        }
        Output<Node> inv_perm;
        if (has_inv) {
            inv_perm = pm.at(extra_inv_argsort_m).get_node_shared_ptr()->output(1);
        }
        // MoEExpertSort is stable, while the argsort TopK may order the rows of equal ids differently,
        // e.g. for ids [1, 0, 1] TopK can give perm [1, 2, 0] and MoEExpertSort gives [1, 0, 2].
        // So all the users of perm must move to MoEExpertSort, or they would disagree on the order of the rows:
        //
        //                       perm (TopK)
        //         ┌──────────────┼──────────────┬──────────────────┐
        //     sorted_ids       token         inv_perm         other user
        //         └─── replaced by MoEExpertSort ───┘    keeps the TopK order
        //
        // ShapeOf reads only the number of rows, so it may stay.
        const std::unordered_set<const Node*> replaced_users{pm.at(sorted_ids_m).get_node(), perm.get_node(), token.get_node(), inv_perm.get_node()};
        auto has_unreplaced_user = [&](const Output<Node>& permutation) {
            const auto users = permutation.get_target_inputs();
            return std::any_of(users.begin(), users.end(), [&](const Input<Node>& user) {
                const auto* node = user.get_node();
                return replaced_users.count(node) == 0U && !ov::is_type_any_of<v0::ShapeOf, v3::ShapeOf>(node);
            });
        };
        if (has_unreplaced_user(perm_out) || has_unreplaced_user(perm)) {
            return false;
        }

        auto sort = std::make_shared<ov::intel_gpu::op::MoEExpertSort>(topk_ids, num_experts);
        sort->set_friendly_name(offsets.get_node()->get_friendly_name());
        NodeVector replaced{perm_out.get_node_shared_ptr(), offsets.get_node_shared_ptr()};
        if (has_token) {
            replaced.push_back(token.get_node_shared_ptr());
        }
        if (has_inv) {
            replaced.push_back(inv_perm.get_node_shared_ptr());
        }
        ov::copy_runtime_info(replaced, sort);

        auto replace = [](Output<Node> old_out, const Output<Node>& new_out) {
            if (old_out.get_element_type() == new_out.get_element_type()) {
                old_out.replace(new_out);
                return;
            }
            // MoEExpertSort outputs i32, while the replaced node can be i64 (ConvertPrecision to i32 runs later): keep the consumers' type
            auto convert = std::make_shared<v0::Convert>(new_out, old_out.get_element_type());
            ov::copy_runtime_info(old_out.get_node_shared_ptr(), convert);
            old_out.replace(convert);
        };
        if (has_token) {
            replace(token, sort->output(0));
        }
        replace(offsets, sort->output(1));
        if (has_inv) {
            replace(inv_perm, sort->output(2));
        }
        return true;
    };

    auto m = std::make_shared<Matcher>(offsets_m, "FuseMoEExpertSort");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
