// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/eliminate_gratuitous_slice_cascade.hpp"

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;

namespace {

// Returns true iff `node` is a v0::Constant whose values are all `false`.
bool is_all_false_bool_constant(const std::shared_ptr<ov::Node>& node) {
    auto cnst = ov::as_type_ptr<v0::Constant>(node);
    if (!cnst || cnst->get_element_type() != ov::element::boolean) {
        return false;
    }
    const auto vals = cnst->get_vector<bool>();
    if (vals.empty()) {
        return false;
    }
    return std::none_of(vals.begin(), vals.end(), [](bool v) {
        return v;
    });
}

// Returns true iff `node` is a v0::Constant whose values are all 0 (integer types).
bool is_all_zero_int_constant(const std::shared_ptr<ov::Node>& node) {
    auto cnst = ov::as_type_ptr<v0::Constant>(node);
    if (!cnst) {
        return false;
    }
    const auto vals = cnst->cast_vector<int64_t>();
    if (vals.empty()) {
        return false;
    }
    return std::all_of(vals.begin(), vals.end(), [](int64_t v) {
        return v == 0;
    });
}

// Returns true iff `node` is `Less(size_const, zero_const)` where size_const has only
// non-negative values and zero_const is all zeros — i.e. the original
// `negative_sizes_mask = Less(size, 0)` produced by translate_slice_op, in the case
// where the Slice's `size` is statically known to be non-negative.
bool is_less_with_all_nonneg_size(const std::shared_ptr<ov::Node>& node) {
    auto less = ov::as_type_ptr<v1::Less>(node);
    if (!less) {
        return false;
    }
    auto lhs = ov::as_type_ptr<v0::Constant>(less->input_value(0).get_node_shared_ptr());
    auto rhs = ov::as_type_ptr<v0::Constant>(less->input_value(1).get_node_shared_ptr());
    if (!lhs || !rhs) {
        return false;
    }
    if (!is_all_zero_int_constant(rhs)) {
        return false;
    }
    const auto lhs_vals = lhs->cast_vector<int64_t>();
    if (lhs_vals.empty()) {
        return false;
    }
    return std::all_of(lhs_vals.begin(), lhs_vals.end(), [](int64_t v) {
        return v >= 0;
    });
}

// Accept the two forms of the Select's condition input that the matcher can fold:
//   - a pre-folded boolean Constant (TransposeSinking's inner CF already collapsed `Less(size, 0)`)
//   - a live `Less(size_const, zero_const)` with non-negative size
// Both must evaluate to all-false so the Select picks the else branch.
bool is_acceptable_cascade_condition(const std::shared_ptr<ov::Node>& cond) {
    return is_all_false_bool_constant(cond) || is_less_with_all_nonneg_size(cond);
}

// The Select's else branch in the gratuitous cascade is either:
//   - a pre-folded v0::Constant (Add(start_const, size_const) already collapsed), or
//   - a live `v1::Add(v0::Constant, v0::Constant)` (translator output before CF).
// Returned shape carries exactly one of these forms.
struct ElseBranch {
    std::shared_ptr<v0::Constant> folded;   // set iff already a single Constant
    std::shared_ptr<v1::Add> add;           // set iff Add(C, C)
    std::shared_ptr<v0::Constant> start_n;  // set iff Add(C, C)
    std::shared_ptr<v0::Constant> size_n;   // set iff Add(C, C)
};

std::optional<ElseBranch> try_extract_else_branch(const std::shared_ptr<ov::Node>& node) {
    if (auto folded = ov::as_type_ptr<v0::Constant>(node)) {
        return ElseBranch{folded, nullptr, nullptr, nullptr};
    }
    auto add = ov::as_type_ptr<v1::Add>(node);
    if (!add) {
        return std::nullopt;
    }
    auto start_n = ov::as_type_ptr<v0::Constant>(add->input_value(0).get_node_shared_ptr());
    auto size_n = ov::as_type_ptr<v0::Constant>(add->input_value(1).get_node_shared_ptr());
    if (!start_n || !size_n) {
        return std::nullopt;
    }
    return ElseBranch{nullptr, add, start_n, size_n};
}

}  // namespace

ov::pass::EliminateGratuitousSliceCascade::EliminateGratuitousSliceCascade() {
    MATCHER_SCOPE(EliminateGratuitousSliceCascade);

    auto data_pat = pattern::any_input();
    auto shape_of_pat = pattern::wrap_type<v3::ShapeOf>({data_pat});
    auto cvtlike_pat = pattern::wrap_type<v1::ConvertLike>({shape_of_pat, pattern::any_input()});
    auto select_pat = pattern::wrap_type<v1::Select>({pattern::any_input(), cvtlike_pat, pattern::any_input()});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto select_node = ov::as_type_ptr<v1::Select>(pattern_map.at(select_pat).get_node_shared_ptr());
        if (!select_node) {
            return false;
        }

        if (!is_acceptable_cascade_condition(select_node->input_value(0).get_node_shared_ptr())) {
            return false;
        }

        auto else_opt = try_extract_else_branch(select_node->input_value(2).get_node_shared_ptr());
        if (!else_opt) {
            return false;
        }

        // Pre-folded Constant — replace Select directly.
        if (else_opt->folded) {
            return ov::replace_output_update_name(select_node->output(0), else_opt->folded->output(0));
        }

        // Live Add(C, C) — synthesise a single Constant carrying start + size.
        // Emitting a Constant (not just `Add(C, C)`) makes the downstream Slice statically
        // inferable even in pipelines that do not run a follow-up ConstantFolding pass
        // (notably NPUW's pre-partition stage).
        const auto start_vals = else_opt->start_n->cast_vector<int64_t>();
        const auto size_vals = else_opt->size_n->cast_vector<int64_t>();
        if (start_vals.empty() || start_vals.size() != size_vals.size()) {
            return false;
        }
        const auto add_out_pshape = else_opt->add->get_output_partial_shape(0);
        if (add_out_pshape.is_dynamic()) {
            return false;
        }

        std::vector<int64_t> stop_vals(start_vals.size());
        std::transform(start_vals.begin(),
                       start_vals.end(),
                       size_vals.begin(),
                       stop_vals.begin(),
                       std::plus<int64_t>());

        auto new_constant =
            std::make_shared<v0::Constant>(else_opt->add->get_element_type(), add_out_pshape.get_shape(), stop_vals);
        ov::copy_runtime_info({select_node, else_opt->add, else_opt->start_n, else_opt->size_n}, new_constant);
        return ov::replace_output_update_name(select_node->output(0), new_constant->output(0));
    };

    auto m = std::make_shared<pattern::Matcher>(select_pat, matcher_name);
    register_matcher(m, callback);
}
