// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/ngram_hash_function_fusion.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/bitwise_xor.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/ngram_hash_function.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

// A named enclosing namespace keeps these generic helper names unique when UNITY_BUILD combines sources.
namespace ov::pass::ngram_hash_function_fusion {
namespace {

// One (token, multiplier) term of the XOR mix.
struct Term {
    ov::Output<ov::Node> token;
    int64_t multiplier;
    bool operator==(const Term& other) const {
        return token == other.token && multiplier == other.multiplier;
    }
};

// A constant broadcast along the last axis only: scalar, [H] or [1, ..., 1, H].
bool is_last_axis_constant(const ov::Output<ov::Node>& output) {
    const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(output.get_node_shared_ptr());
    if (!constant) {
        return false;
    }
    const auto& shape = constant->get_shape();
    return shape.empty() || ov::shape_size(shape) == shape.back();
}

std::vector<int64_t> constant_values(const ov::Output<ov::Node>& output) {
    return ov::as_type_ptr<ov::op::v0::Constant>(output.get_node_shared_ptr())->cast_vector<int64_t>();
}

// Unsqueeze(x, -1) or the equivalent Reshape produced by CommonOptimizations.
bool appends_unit_last_dim(const ov::Output<ov::Node>& output) {
    const auto node = output.get_node_shared_ptr();
    const auto& in_shape = node->get_input_partial_shape(0);
    const auto& out_shape = node->get_output_partial_shape(0);
    if (in_shape.rank().is_dynamic() || out_shape.rank().is_dynamic() || out_shape.size() != in_shape.size() + 1 ||
        ov::op::util::get_unsqueeze_axes(in_shape, out_shape) !=
            std::vector<int64_t>{static_cast<int64_t>(in_shape.size())}) {
        return false;
    }
    auto expected_shape = in_shape;
    expected_shape.push_back(1);
    if (!out_shape.same_scheme(expected_shape)) {
        return false;
    }
    if (ov::as_type_ptr<ov::op::v0::Unsqueeze>(node)) {
        return true;  // Unsqueeze passes every input dimension through, so the shapes are enough
    }
    // Shapes alone cannot tell whether a Reshape keeps dynamic dimensions, so its target shape must copy them.
    if (const auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(node)) {
        const auto pattern = ov::as_type_ptr<ov::op::v0::Constant>(node->get_input_node_shared_ptr(1));
        if (pattern) {
            const auto values = pattern->cast_vector<int64_t>();
            if (values.size() != in_shape.size() + 1 || values.back() != 1) {
                return false;
            }
            // A literal zero makes '-1' ambiguous: a zero-sized output cannot tell which input dimension it replaces.
            const bool infers_dim = std::find(values.begin(), values.end(), -1) != values.end();
            if (infers_dim && !reshape->get_special_zero() &&
                std::find(values.begin(), values.end(), 0) != values.end()) {
                return false;
            }
            for (size_t i = 0; i < in_shape.size(); ++i) {
                if (values[i] == -1) {
                    // The single inferred dimension is preserved if every other prefix dimension is copied.
                    continue;
                }
                if (reshape->get_special_zero() && values[i] == 0) {
                    continue;
                }
                if (in_shape[i].is_dynamic() || out_shape[i].is_dynamic() || in_shape[i] != out_shape[i]) {
                    return false;
                }
            }
            return true;
        }
        const auto concat = ov::as_type_ptr<ov::op::v0::Concat>(node->get_input_node_shared_ptr(1));
        if (!concat || concat->get_axis() != 0 || concat->get_input_size() != 2) {
            return false;
        }
        const auto shape_of = ov::as_type_ptr<ov::op::util::ShapeOfBase>(concat->get_input_node_shared_ptr(0));
        const auto one = ov::as_type_ptr<ov::op::v0::Constant>(concat->get_input_node_shared_ptr(1));
        return shape_of && shape_of->input_value(0) == node->input_value(0) && one &&
               one->get_shape() == ov::Shape{1} && one->cast_vector<int64_t>() == std::vector<int64_t>{1};
    }
    return false;
}

bool is_last_axis(int64_t axis, const ov::Rank& rank) {
    return axis == -1 || (rank.is_static() && axis == rank.get_length() - 1);
}

using MultiplierPredicate = std::function<bool(const ov::Output<ov::Node>&)>;

// Walks the XOR tree of the mix; its depth depends on the n-gram order, so it cannot be a fixed pattern.
bool collect_terms(const ov::Output<ov::Node>& output,
                   const MultiplierPredicate& is_multiplier,
                   std::vector<Term>& terms,
                   ov::NodeVector& nodes) {
    const auto node = output.get_node_shared_ptr();
    if (const auto xor_node = ov::as_type_ptr<ov::op::v13::BitwiseXor>(node)) {
        if (xor_node->get_autob().m_type != ov::op::AutoBroadcastType::NUMPY) {
            return false;
        }
        nodes.push_back(xor_node);
        return collect_terms(xor_node->input_value(0), is_multiplier, terms, nodes) &&
               collect_terms(xor_node->input_value(1), is_multiplier, terms, nodes);
    }
    const auto mul_node = ov::as_type_ptr<ov::op::v1::Multiply>(node);
    if (!mul_node || mul_node->get_autob().m_type != ov::op::AutoBroadcastType::NUMPY) {
        return false;
    }
    const bool lhs_const = is_multiplier(mul_node->input_value(0));
    if (lhs_const == is_multiplier(mul_node->input_value(1))) {
        return false;  // exactly one scalar constant side
    }
    Term term{mul_node->input_value(lhs_const ? 1 : 0), constant_values(mul_node->input_value(lhs_const ? 0 : 1))[0]};
    const auto& token_rank = term.token.get_partial_shape().rank();
    const auto multiplier_rank = mul_node->get_input_shape(lhs_const ? 0 : 1).size();
    if (token_rank.is_dynamic() || multiplier_rank > static_cast<size_t>(token_rank.get_length())) {
        return false;
    }
    if (term.token.get_element_type() != ov::element::i64 ||
        std::find(terms.begin(), terms.end(), term) != terms.end()) {
        return false;  // x ^ x cancels out, not an n-gram hash
    }
    nodes.push_back(mul_node);
    terms.push_back(std::move(term));
    return true;
}

bool rows_fit_int32(const std::vector<int64_t>& vocab_sizes, const std::vector<int64_t>& offsets) {
    for (size_t h = 0; h < vocab_sizes.size(); ++h) {
        if (vocab_sizes[h] <= 0 || offsets[h] < 0 ||
            offsets[h] > std::numeric_limits<int32_t>::max() - (vocab_sizes[h] - 1)) {
            return false;
        }
    }
    return true;
}

// Replaces `root` by Convert(NGramHashFunction(Concat(Unsqueeze(x[n-1]), ..., Unsqueeze(x[0])))).
bool replace_with_hash(const std::shared_ptr<ov::Node>& root,
                       const std::vector<Term>& terms,
                       const std::vector<int64_t>& heads_per_order,
                       const std::vector<int64_t>& vocab_sizes,
                       const std::vector<int64_t>& offsets,
                       const ov::NodeVector& matched_nodes) {
    // Tokens are packed without broadcasting: tokens that only broadcast to a common shape are rejected here when
    // static, and make the packing Concat fail at runtime when dynamic, instead of silently changing results.
    auto token_shape = terms[0].token.get_partial_shape();
    for (const auto& term : terms) {
        if (!ov::PartialShape::merge_into(token_shape, term.token.get_partial_shape())) {
            return false;
        }
    }
    const auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    ov::NodeVector new_nodes{axis};
    ov::OutputVector window;
    for (auto it = terms.rbegin(); it != terms.rend(); ++it) {
        window.push_back(std::make_shared<ov::op::v0::Unsqueeze>(it->token, axis));
    }
    std::vector<int64_t> multipliers;
    for (const auto& term : terms) {
        multipliers.push_back(term.multiplier);
    }
    const auto ngrams = std::make_shared<ov::op::v0::Concat>(window, -1);
    const auto hash = std::make_shared<ov::op::internal::NGramHashFunction>(ngrams,
                                                                            heads_per_order,
                                                                            multipliers,
                                                                            vocab_sizes,
                                                                            offsets);
    if (!hash->get_output_partial_shape(0).compatible(root->get_output_partial_shape(0))) {
        return false;
    }
    const auto convert = std::make_shared<ov::op::v0::Convert>(hash, root->get_output_element_type(0));
    hash->set_friendly_name(root->get_friendly_name() + "/NGramHashFunction");
    convert->set_friendly_name(root->get_friendly_name());
    new_nodes.insert(new_nodes.end(), {ngrams, hash, convert});
    for (const auto& w : window) {
        new_nodes.push_back(w.get_node_shared_ptr());
    }
    ov::copy_runtime_info(matched_nodes, new_nodes);
    ov::replace_node(root, convert);
    return true;
}

// A single-order NGramHashFunction produced by NGramHashBlockFusion, behind its i64 Convert.
std::shared_ptr<ov::op::internal::NGramHashFunction> get_single_order_hash(const ov::Output<ov::Node>& output) {
    const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(output.get_node_shared_ptr());
    if (!convert || convert->get_output_element_type(0) != ov::element::i64) {
        return nullptr;
    }
    const auto hash = ov::as_type_ptr<ov::op::internal::NGramHashFunction>(convert->get_input_node_shared_ptr(0));
    if (!hash) {
        return nullptr;
    }
    const auto& heads = hash->get_heads_per_order();
    const bool single_order = std::all_of(heads.begin(), heads.end() - 1, [](int64_t h) {
        return h == 0;
    });
    return single_order ? hash : nullptr;
}

// The (token, multiplier) terms of a packed window Concat(Unsqueeze(x[n-1]), ..., Unsqueeze(x[0])).
std::optional<std::vector<Term>> get_terms(const ov::op::internal::NGramHashFunction& hash) {
    const auto window = ov::as_type_ptr<ov::op::v0::Concat>(hash.get_input_node_shared_ptr(0));
    const auto n = static_cast<size_t>(hash.get_ngram_size());
    if (!window || window->get_input_size() != n ||
        !is_last_axis(window->get_axis(), window->get_output_partial_shape(0).rank())) {
        return std::nullopt;
    }
    std::vector<Term> terms;
    for (size_t k = 0; k < n; ++k) {
        const auto unsqueeze = window->input_value(n - 1 - k);
        if (!ov::as_type_ptr<ov::op::v0::Unsqueeze>(unsqueeze.get_node_shared_ptr()) ||
            !appends_unit_last_dim(unsqueeze)) {
            return std::nullopt;
        }
        Term term{unsqueeze.get_node()->input_value(0), hash.get_multipliers()[k]};
        if (std::find(terms.begin(), terms.end(), term) != terms.end()) {
            return std::nullopt;  // x ^ x cancels out; the concat merge treats terms as a set
        }
        terms.push_back(std::move(term));
    }
    return terms;
}

}  // namespace
}  // namespace ov::pass::ngram_hash_function_fusion

ov::pass::NGramHashBlockFusion::NGramHashBlockFusion() {
    MATCHER_SCOPE(NGramHashBlockFusion);
    // Exporters keep the hash multipliers as scalars or as [1] / [1, 1] constants.
    const auto scalar_multiplier = ov::pass::pattern::shape_matches("[]") || ov::pass::pattern::shape_matches("[1]") ||
                                   ov::pass::pattern::shape_matches("[1, 1]");

    // mix = ... ^ (token * multiplier): the top XOR has at least one Multiply-by-scalar operand.
    auto term_m = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>(
        {ov::pass::pattern::any_input(ov::pass::pattern::type_matches(ov::element::i64)),
         ov::pass::pattern::wrap_type<ov::op::v0::Constant>(scalar_multiplier)},
        ov::pass::pattern::type_matches(ov::element::i64),
        {{"auto_broadcast", "numpy"}});
    // BitwiseXor is not commutative for the matcher (ov::op::util::is_commutative), so both operand orders are listed.
    auto xor_m =
        ov::pass::pattern::wrap_type<ov::op::v13::BitwiseXor>({ov::pass::pattern::any_input(), term_m},
                                                              ov::pass::pattern::type_matches(ov::element::i64),
                                                              {{"auto_broadcast", "numpy"}}) |
        ov::pass::pattern::wrap_type<ov::op::v13::BitwiseXor>({term_m, ov::pass::pattern::any_input()},
                                                              ov::pass::pattern::type_matches(ov::element::i64),
                                                              {{"auto_broadcast", "numpy"}});
    auto unsqueeze_m = ov::pass::pattern::wrap_type<ov::op::v0::Unsqueeze, ov::op::v1::Reshape>(
        {xor_m, ov::pass::pattern::any_input()},
        ngram_hash_function_fusion::appends_unit_last_dim);
    auto vocab_m =
        ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ngram_hash_function_fusion::is_last_axis_constant);
    auto mod_m = ov::pass::pattern::wrap_type<ov::op::v1::FloorMod>({unsqueeze_m, vocab_m},
                                                                    ov::pass::pattern::type_matches(ov::element::i64),
                                                                    {{"auto_broadcast", "numpy"}});
    auto offsets_m =
        ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ngram_hash_function_fusion::is_last_axis_constant);
    auto add_m = ov::pass::pattern::wrap_type<ov::op::v1::Add>({mod_m, offsets_m},
                                                               ov::pass::pattern::type_matches(ov::element::i64),
                                                               {{"auto_broadcast", "numpy"}});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pm = m.get_pattern_value_map();
        const auto root = m.get_match_root();
        if (transformation_callback(root)) {
            return false;
        }
        const auto unsqueeze = pm.at(unsqueeze_m).get_node_shared_ptr();

        auto vocab_sizes = ngram_hash_function_fusion::constant_values(pm.at(vocab_m));
        auto offsets = ngram_hash_function_fusion::constant_values(pm.at(offsets_m));
        if (offsets.size() == 1) {
            offsets.resize(vocab_sizes.size(), offsets[0]);
        }
        if (vocab_sizes.empty() || offsets.size() != vocab_sizes.size() ||
            !ngram_hash_function_fusion::rows_fit_int32(vocab_sizes, offsets)) {
            return false;
        }

        // The deeper XOR levels are walked manually, so their multipliers are checked with the same shape predicate.
        const auto is_multiplier = [&](const ov::Output<ov::Node>& output) {
            return ov::as_type_ptr<ov::op::v0::Constant>(output.get_node_shared_ptr()) && scalar_multiplier(&m, output);
        };
        std::vector<ngram_hash_function_fusion::Term> terms;
        ov::NodeVector matched_nodes{root, pm.at(mod_m).get_node_shared_ptr(), unsqueeze};
        if (!ngram_hash_function_fusion::collect_terms(unsqueeze->input_value(0),
                                                       is_multiplier,
                                                       terms,
                                                       matched_nodes) ||
            terms.size() < 2) {
            return false;
        }
        std::vector<int64_t> heads_per_order(terms.size() - 1, 0);
        heads_per_order.back() = static_cast<int64_t>(vocab_sizes.size());
        return ngram_hash_function_fusion::replace_with_hash(root,
                                                             terms,
                                                             heads_per_order,
                                                             vocab_sizes,
                                                             offsets,
                                                             matched_nodes);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(add_m, matcher_name);
    register_matcher(m, callback);
}

ov::pass::NGramHashConcatFusion::NGramHashConcatFusion() {
    MATCHER_SCOPE(NGramHashConcatFusion);
    auto concat_m = ov::pass::pattern::wrap_type<ov::op::v0::Concat>([](const ov::Output<ov::Node>& output) {
        const auto concat = ov::as_type_ptr<ov::op::v0::Concat>(output.get_node_shared_ptr());
        const auto& inputs = concat->input_values();
        return output.get_element_type() == ov::element::i64 && inputs.size() > 1 &&
               ngram_hash_function_fusion::is_last_axis(concat->get_axis(), output.get_partial_shape().rank()) &&
               std::all_of(inputs.begin(), inputs.end(), [](const ov::Output<ov::Node>& input) {
                   return ngram_hash_function_fusion::get_single_order_hash(input) != nullptr;
               });
    });

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto root = m.get_match_root();
        if (transformation_callback(root)) {
            return false;
        }
        // Every order must extend the terms of the previous one; XOR is commutative, so only the sets matter.
        std::vector<ngram_hash_function_fusion::Term> terms;
        std::vector<int64_t> heads_per_order;
        std::vector<int64_t> vocab_sizes;
        std::vector<int64_t> offsets;
        ov::NodeVector matched_nodes{root};
        for (const auto& input : root->input_values()) {
            const auto hash = ngram_hash_function_fusion::get_single_order_hash(input);
            const auto order_terms = ngram_hash_function_fusion::get_terms(*hash);
            if (!order_terms || order_terms->size() <= terms.size()) {
                return false;  // orders must be strictly increasing along the concatenated heads
            }
            for (const auto& term : terms) {
                if (std::find(order_terms->begin(), order_terms->end(), term) == order_terms->end()) {
                    return false;
                }
            }
            for (const auto& term : *order_terms) {
                if (std::find(terms.begin(), terms.end(), term) == terms.end()) {
                    terms.push_back(term);
                }
            }
            heads_per_order.resize(terms.size() - 1, 0);
            heads_per_order.back() = hash->get_heads_per_order().back();
            vocab_sizes.insert(vocab_sizes.end(),
                               hash->get_head_vocab_sizes().begin(),
                               hash->get_head_vocab_sizes().end());
            offsets.insert(offsets.end(), hash->get_head_offsets().begin(), hash->get_head_offsets().end());
            matched_nodes.insert(matched_nodes.end(),
                                 {input.get_node_shared_ptr(), hash, hash->get_input_node_shared_ptr(0)});
        }
        return ngram_hash_function_fusion::replace_with_hash(root,
                                                             terms,
                                                             heads_per_order,
                                                             vocab_sizes,
                                                             offsets,
                                                             matched_nodes);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(concat_m, matcher_name);
    register_matcher(m, callback);
}
