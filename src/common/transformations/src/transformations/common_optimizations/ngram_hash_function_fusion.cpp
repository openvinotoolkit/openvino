// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/ngram_hash_function_fusion.hpp"

#include <algorithm>
#include <cstdint>
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
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov::op;
using ov::op::internal::NGramHashFunction;
using ov::pass::pattern::any_input;
using ov::pass::pattern::type_matches;
using ov::pass::pattern::wrap_type;

namespace {

// One (token, multiplier) term of the XOR mix.
struct Term {
    ov::Output<ov::Node> token;
    int64_t multiplier;
    bool operator==(const Term& other) const {
        return token == other.token && multiplier == other.multiplier;
    }
};

bool is_scalar_constant(const ov::Output<ov::Node>& output) {
    const auto constant = ov::as_type_ptr<v0::Constant>(output.get_node_shared_ptr());
    return constant && ov::shape_size(constant->get_shape()) == 1;
}

// A constant broadcast along the last axis only: scalar, [H] or [1, ..., 1, H].
bool is_last_axis_constant(const ov::Output<ov::Node>& output) {
    const auto constant = ov::as_type_ptr<v0::Constant>(output.get_node_shared_ptr());
    if (!constant) {
        return false;
    }
    const auto& shape = constant->get_shape();
    return shape.empty() || ov::shape_size(shape) == shape.back();
}

std::vector<int64_t> constant_values(const ov::Output<ov::Node>& output) {
    return ov::as_type_ptr<v0::Constant>(output.get_node_shared_ptr())->cast_vector<int64_t>();
}

// Unsqueeze(x, -1) or the equivalent Reshape produced by CommonOptimizations.
bool appends_unit_last_dim(const ov::Output<ov::Node>& output) {
    const auto node = output.get_node_shared_ptr();
    const auto& in_shape = node->get_input_partial_shape(0);
    if (ov::as_type_ptr<v0::Unsqueeze>(node)) {
        const auto axes = ov::as_type_ptr<v0::Constant>(node->get_input_node_shared_ptr(1));
        if (!axes) {
            return false;
        }
        const auto values = axes->cast_vector<int64_t>();
        return values.size() == 1 &&
               (values[0] == -1 || (in_shape.rank().is_static() && values[0] == in_shape.rank().get_length()));
    }
    if (ov::as_type_ptr<v1::Reshape>(node)) {
        const auto& out_shape = node->get_output_partial_shape(0);
        if (in_shape.rank().is_dynamic() || out_shape.rank().is_dynamic() ||
            out_shape.size() != in_shape.size() + 1 || out_shape[out_shape.size() - 1] != 1) {
            return false;
        }
        for (size_t i = 0; i < in_shape.size(); ++i) {
            if (!in_shape[i].compatible(out_shape[i])) {
                return false;
            }
        }
        return true;
    }
    return false;
}

bool is_last_axis(int64_t axis, const ov::Rank& rank) {
    return axis == -1 || (rank.is_static() && axis == rank.get_length() - 1);
}

// Walks the XOR tree of the mix; its depth depends on the n-gram order, so it cannot be a fixed pattern.
bool collect_terms(const ov::Output<ov::Node>& output, std::vector<Term>& terms, ov::NodeVector& nodes) {
    const auto node = output.get_node_shared_ptr();
    if (const auto xor_node = ov::as_type_ptr<v13::BitwiseXor>(node)) {
        nodes.push_back(xor_node);
        return collect_terms(xor_node->input_value(0), terms, nodes) &&
               collect_terms(xor_node->input_value(1), terms, nodes);
    }
    const auto mul_node = ov::as_type_ptr<v1::Multiply>(node);
    if (!mul_node) {
        return false;
    }
    const bool lhs_const = is_scalar_constant(mul_node->input_value(0));
    if (lhs_const == is_scalar_constant(mul_node->input_value(1))) {
        return false;  // exactly one scalar constant side
    }
    Term term{mul_node->input_value(lhs_const ? 1 : 0), constant_values(mul_node->input_value(lhs_const ? 0 : 1))[0]};
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
    auto token_shape = terms[0].token.get_partial_shape();
    for (const auto& term : terms) {
        if (!ov::PartialShape::merge_into(token_shape, term.token.get_partial_shape())) {
            return false;
        }
    }
    const auto axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    ov::OutputVector window;
    for (auto it = terms.rbegin(); it != terms.rend(); ++it) {
        window.push_back(std::make_shared<v0::Unsqueeze>(it->token, axis));
    }
    std::vector<int64_t> multipliers;
    for (const auto& term : terms) {
        multipliers.push_back(term.multiplier);
    }
    const auto ngrams = std::make_shared<v0::Concat>(window, -1);
    const auto hash = std::make_shared<NGramHashFunction>(ngrams,
                                                          static_cast<int64_t>(terms.size()),
                                                          heads_per_order,
                                                          multipliers,
                                                          vocab_sizes,
                                                          offsets);
    if (!hash->get_output_partial_shape(0).compatible(root->get_output_partial_shape(0))) {
        return false;
    }
    const auto convert = std::make_shared<v0::Convert>(hash, root->get_output_element_type(0));
    hash->set_friendly_name(root->get_friendly_name() + "/NGramHashFunction");
    convert->set_friendly_name(root->get_friendly_name());
    ov::NodeVector new_nodes{ngrams, hash, convert};
    for (const auto& w : window) {
        new_nodes.push_back(w.get_node_shared_ptr());
    }
    ov::copy_runtime_info(matched_nodes, new_nodes);
    ov::replace_node(root, convert);
    return true;
}

// A single-order NGramHashFunction produced by NGramHashBlockFusion, behind its i64 Convert.
std::shared_ptr<NGramHashFunction> get_single_order_hash(const ov::Output<ov::Node>& output) {
    const auto convert = ov::as_type_ptr<v0::Convert>(output.get_node_shared_ptr());
    if (!convert || convert->get_output_element_type(0) != ov::element::i64) {
        return nullptr;
    }
    const auto hash = ov::as_type_ptr<NGramHashFunction>(convert->get_input_node_shared_ptr(0));
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
std::optional<std::vector<Term>> get_terms(const NGramHashFunction& hash) {
    const auto window = ov::as_type_ptr<v0::Concat>(hash.get_input_node_shared_ptr(0));
    const auto n = static_cast<size_t>(hash.get_ngram_size());
    if (!window || window->get_input_size() != n ||
        !is_last_axis(window->get_axis(), window->get_output_partial_shape(0).rank())) {
        return std::nullopt;
    }
    std::vector<Term> terms;
    for (size_t k = 0; k < n; ++k) {
        const auto unsqueeze = window->input_value(n - 1 - k);
        if (!ov::as_type_ptr<v0::Unsqueeze>(unsqueeze.get_node_shared_ptr()) || !appends_unit_last_dim(unsqueeze)) {
            return std::nullopt;
        }
        terms.push_back({unsqueeze.get_node()->input_value(0), hash.get_multipliers()[k]});
    }
    return terms;
}

}  // namespace

ov::pass::NGramHashBlockFusion::NGramHashBlockFusion() {
    MATCHER_SCOPE(NGramHashBlockFusion);
    // mix = ... ^ (token * multiplier): the top XOR has at least one Multiply-by-scalar operand.
    auto term_m = wrap_type<v1::Multiply>({any_input(type_matches(element::i64)), wrap_type<v0::Constant>(is_scalar_constant)});
    auto xor_m = std::make_shared<ov::pass::pattern::op::Or>(
        OutputVector{wrap_type<v13::BitwiseXor>({any_input(), term_m}, type_matches(element::i64)),
                     wrap_type<v13::BitwiseXor>({term_m, any_input()}, type_matches(element::i64))});
    auto unsqueeze_m = wrap_type<v0::Unsqueeze, v1::Reshape>({xor_m, any_input()}, appends_unit_last_dim);
    auto vocab_m = wrap_type<v0::Constant>(is_last_axis_constant);
    auto mod_m = wrap_type<v1::FloorMod>({unsqueeze_m, vocab_m});
    auto offsets_m = wrap_type<v0::Constant>(is_last_axis_constant);
    auto add_m = wrap_type<v1::Add>({mod_m, offsets_m}, type_matches(element::i64));

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pm = m.get_pattern_value_map();
        const auto root = m.get_match_root();
        const auto unsqueeze = pm.at(unsqueeze_m).get_node_shared_ptr();

        auto vocab_sizes = constant_values(pm.at(vocab_m));
        auto offsets = constant_values(pm.at(offsets_m));
        if (offsets.size() == 1) {
            offsets.resize(vocab_sizes.size(), offsets[0]);
        }
        if (vocab_sizes.empty() || offsets.size() != vocab_sizes.size() || !rows_fit_int32(vocab_sizes, offsets)) {
            return false;
        }

        std::vector<Term> terms;
        NodeVector matched_nodes{root, pm.at(mod_m).get_node_shared_ptr(), unsqueeze};
        if (!collect_terms(unsqueeze->input_value(0), terms, matched_nodes) || terms.size() < 2) {
            return false;
        }
        std::vector<int64_t> heads_per_order(terms.size() - 1, 0);
        heads_per_order.back() = static_cast<int64_t>(vocab_sizes.size());
        return replace_with_hash(root, terms, heads_per_order, vocab_sizes, offsets, matched_nodes);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(add_m, matcher_name);
    register_matcher(m, callback);
}

ov::pass::NGramHashConcatFusion::NGramHashConcatFusion() {
    MATCHER_SCOPE(NGramHashConcatFusion);
    auto concat_m = wrap_type<v0::Concat>([](const Output<Node>& output) {
        const auto concat = ov::as_type_ptr<v0::Concat>(output.get_node_shared_ptr());
        const auto& inputs = concat->input_values();
        return output.get_element_type() == element::i64 && inputs.size() > 1 &&
               is_last_axis(concat->get_axis(), output.get_partial_shape().rank()) &&
               std::all_of(inputs.begin(), inputs.end(), [](const Output<Node>& input) {
                   return get_single_order_hash(input) != nullptr;
               });
    });

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto root = m.get_match_root();
        // Every order must extend the terms of the previous one; XOR is commutative, so only the sets matter.
        std::vector<Term> terms;
        std::vector<int64_t> heads_per_order;
        std::vector<int64_t> vocab_sizes;
        std::vector<int64_t> offsets;
        NodeVector matched_nodes{root};
        for (const auto& input : root->input_values()) {
            const auto hash = get_single_order_hash(input);
            const auto order_terms = get_terms(*hash);
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
            vocab_sizes.insert(vocab_sizes.end(), hash->get_head_vocab_sizes().begin(), hash->get_head_vocab_sizes().end());
            offsets.insert(offsets.end(), hash->get_head_offsets().begin(), hash->get_head_offsets().end());
            matched_nodes.insert(matched_nodes.end(),
                                 {input.get_node_shared_ptr(), hash, hash->get_input_node_shared_ptr(0)});
        }
        return replace_with_hash(root, terms, heads_per_order, vocab_sizes, offsets, matched_nodes);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(concat_m, matcher_name);
    register_matcher(m, callback);
}
