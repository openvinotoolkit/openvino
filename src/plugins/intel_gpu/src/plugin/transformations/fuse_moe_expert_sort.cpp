// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_moe_expert_sort.hpp"

#include <memory>

#include "intel_gpu/op/moe_expert_sort.hpp"
#include "openvino/core/rt_info.hpp"
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

bool is_zero_constant(const Output<Node>& output) {
    const auto constant = ov::as_type_ptr<v0::Constant>(output.get_node_shared_ptr());
    if (!constant) {
        return false;
    }
    const auto values = constant->cast_vector<int64_t>();
    return std::all_of(values.begin(), values.end(), [](int64_t v) {
        return v == 0;
    });
}

// Returns the output of the first consumer of `out` matched by `pattern`, whose `bound` label is `out`
std::shared_ptr<Matcher> match_consumer(const Output<Node>& out, const std::shared_ptr<Node>& pattern, const std::shared_ptr<Node>& bound) {
    for (const auto& input : out.get_target_inputs()) {
        auto m = std::make_shared<Matcher>(pattern);
        for (const auto& candidate : input.get_node()->outputs()) {
            if (m->match(candidate) && m->get_pattern_value_map().at(bound) == out) {
                return m;
            }
        }
    }
    return nullptr;
}

}  // namespace

FuseMoEExpertSort::FuseMoEExpertSort(size_t max_num_experts) {
    auto topk_ids_m = any_input(rank_equals(2) && has_static_dim(1));
    auto flat_ids_m = wrap_type<v1::Reshape>({topk_ids_m, any_input()}, rank_equals(1));
    auto argsort_m = argsort_pattern(flat_ids_m);
    auto perm_m = optional<v0::Convert>({argsort_m->output(1)});
    auto sorted_ids_m = wrap_type<v7::Gather, v8::Gather>({flat_ids_m, perm_m, 0}, {{"batch_dims", 0}});
    auto sorted_ids_cvt_m = optional<v0::Convert>({sorted_ids_m});
    auto zeros_m = wrap_type<v0::Constant>(rank_equals(1) && has_static_shape() && is_zero_constant);
    auto ones_m = wrap_type<v1::Broadcast, v3::Broadcast>({wrap_type<v0::Constant>(value_matches("1")), any_input()});
    auto updates_m = optional<v8::Slice>({ones_m, any_input(), any_input(), any_input(), any_input()});
    auto counts_m = wrap_type<v12::ScatterElementsUpdate>({zeros_m, sorted_ids_cvt_m, updates_m, 0}, {{"reduction", "sum"}, {"use_init_val", true}});
    auto offsets_m = wrap_type<v0::CumSum>({counts_m, 0}, {{"exclusive", false}, {"reverse", false}});

    // Token of each sorted row: [0..T) expanded to [T, K], flattened and gathered by the permutation
    auto perm_in_m = any_input();
    auto range_m = wrap_type<v4::Range>({0, any_input(), 1});
    auto unsqueeze_m = wrap_type<v0::Unsqueeze>({range_m, any_input()}, shape_matches("?, 1"));
    auto expand_m = wrap_type<v1::Broadcast, v3::Broadcast>({unsqueeze_m, any_input()}, rank_equals(2) && has_static_dim(1));
    auto token_src_m = wrap_type<v1::Reshape>({expand_m, any_input()}, rank_equals(1));
    auto token_m = wrap_type<v7::Gather, v8::Gather>({token_src_m, perm_in_m, 0}, {{"batch_dims", 0}});

    // Inverse permutation: argsort of the permutation
    auto perm_out_m = any_input();
    auto inv_argsort_m = argsort_pattern(perm_out_m);

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        const auto& pm = m.get_pattern_value_map();
        const auto topk_ids = pm.at(topk_ids_m);
        const auto offsets = pm.at(offsets_m);
        const auto perm_out = m.get_pattern_map().at(argsort_m)->output(1);
        const auto perm = pm.at(perm_m);
        const size_t top_k = static_cast<size_t>(topk_ids.get_partial_shape()[1].get_length());
        const size_t num_experts = pm.at(zeros_m).get_shape()[0];
        if (num_experts > max_num_experts) {
            return false;
        }

        auto token_match = match_consumer(perm, token_m, perm_in_m);
        auto inv_match = match_consumer(perm_out, inv_argsort_m, perm_out_m);
        if (!token_match || !inv_match) {
            return false;
        }
        const auto expand = token_match->get_pattern_value_map().at(expand_m);
        if (static_cast<size_t>(expand.get_partial_shape()[1].get_length()) != top_k) {
            return false;
        }
        const auto token = token_match->get_pattern_value_map().at(token_m);
        const auto inv_perm = inv_match->get_pattern_map().at(inv_argsort_m)->output(1);

        // Every user of the permutation must be replaced, since the new sort may order ties differently
        const auto sorted_ids = pm.at(sorted_ids_m).get_node();
        auto replaced_or_shape = [&](const Input<Node>& in) {
            const auto node = in.get_node();
            return node == sorted_ids || node == token.get_node() || node == inv_perm.get_node() || node == perm.get_node() ||
                   ov::is_type_any_of<v0::ShapeOf, v3::ShapeOf>(node);
        };
        for (const auto& out : {perm_out, perm}) {
            const auto inputs = out.get_target_inputs();
            if (!std::all_of(inputs.begin(), inputs.end(), replaced_or_shape)) {
                return false;
            }
        }

        auto sort = std::make_shared<ov::intel_gpu::op::MoEExpertSort>(topk_ids, num_experts);
        sort->set_friendly_name(offsets.get_node()->get_friendly_name());
        ov::copy_runtime_info({perm_out.get_node_shared_ptr(), offsets.get_node_shared_ptr(), token.get_node_shared_ptr(), inv_perm.get_node_shared_ptr()},
                              sort);

        auto replace = [](Output<Node> old_out, const Output<Node>& new_out) {
            if (old_out.get_element_type() == new_out.get_element_type()) {
                old_out.replace(new_out);
                return;
            }
            auto convert = std::make_shared<v0::Convert>(new_out, old_out.get_element_type());
            ov::copy_runtime_info(old_out.get_node_shared_ptr(), convert);
            old_out.replace(convert);
        };
        replace(token, sort->output(0));
        replace(offsets, sort->output(1));
        replace(inv_perm, sort->output(2));
        return true;
    };

    auto m = std::make_shared<Matcher>(offsets_m, "FuseMoEExpertSort");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
