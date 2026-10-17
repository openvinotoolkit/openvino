// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngram_hash_decomposition.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/bitwise_right_shift.hpp"
#include "openvino/op/bitwise_xor.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov::op;

namespace {

// A traced int64 n-gram hash has the shape:
//   offsets + floor_mod(unsqueeze(xor_i(token_i * mult_i), -1), moduli)
// where `mult_i` are per-position scalar Constants. When some `mult_i` doesn't fit int32,
// CPU's ConvertPrecision clamps it to INT32_MAX instead of converting exactly, silently
// corrupting the hash. This rewrites the matched subtree into exact int32/f32 arithmetic on
// 12-bit limbs.
constexpr int64_t LIMB_BITS = 12;
constexpr int64_t LIMB_MASK = (int64_t{1} << LIMB_BITS) - 1;
constexpr size_t SPREAD_LIMBS = 3;  // token (<= ~18 bits) * 12-bit limb spans at most 3 limbs

int64_t bit_length(int64_t value) {
    int64_t len = 0;
    while (value > 0) {
        ++len;
        value >>= 1;
    }
    return len;
}

int64_t pow_mod(int64_t base, int64_t exp, int64_t mod) {
    int64_t result = 1 % mod;
    int64_t b = base % mod;
    while (exp > 0) {
        if (exp & 1) {
            result = (result * b) % mod;
        }
        b = (b * b) % mod;
        exp >>= 1;
    }
    return result;
}

// x mod modulus for 0 <= x < 2**31, int32: float quotient estimate (off by at most 1) plus an
// exact int32 correction. CPU computes int32 FloorMod through float, which is inexact above 2**24.
// CPU int32 comparisons also go through float, so the corrections only ever compare against zero.
ov::Output<ov::Node> make_mod_exact(const ov::Output<ov::Node>& x,
                                     const ov::Output<ov::Node>& modulus,
                                     const ov::Output<ov::Node>& modulus_f32,
                                     ov::NodeVector& new_ops) {
    auto x_f32 = std::make_shared<v0::Convert>(x, ov::element::f32);
    auto div = std::make_shared<v1::Divide>(x_f32, modulus_f32);
    auto floor = std::make_shared<v0::Floor>(div);
    auto q = std::make_shared<v0::Convert>(floor, ov::element::i32);
    auto qm = std::make_shared<v1::Multiply>(q, modulus);
    auto r = std::make_shared<v1::Subtract>(x, qm);

    auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    auto is_neg = std::make_shared<v1::Less>(r, zero);
    auto r_plus = std::make_shared<v1::Add>(r, modulus);
    auto r_low = std::make_shared<v1::Select>(is_neg, r_plus, r);

    auto r_minus = std::make_shared<v1::Subtract>(r_low, modulus);
    auto is_ge = std::make_shared<v1::GreaterEqual>(r_minus, zero);
    auto result = std::make_shared<v1::Select>(is_ge, r_minus, r_low);

    new_ops.insert(new_ops.end(),
                   {x_f32, div, floor, q, qm, r, zero, is_neg, r_plus, r_low, is_ge, r_minus, result});
    return result;
}

// Exact `token * multiplier` as carry-propagated little-endian 12-bit int32 limbs, all values
// well under 2**31.
std::vector<ov::Output<ov::Node>> build_clean_limbs(const ov::Output<ov::Node>& token,
                                                     int64_t multiplier,
                                                     size_t num_mult_limbs,
                                                     ov::NodeVector& new_ops) {
    auto token_i32 = std::make_shared<v0::Convert>(token, ov::element::i32);
    new_ops.push_back(token_i32);

    const size_t num_clean_limbs = num_mult_limbs + SPREAD_LIMBS;
    auto mask_const = v0::Constant::create(ov::element::i32, ov::Shape{}, {static_cast<int32_t>(LIMB_MASK)});
    auto shift12 = v0::Constant::create(ov::element::i32, ov::Shape{}, {12});
    auto shift24 = v0::Constant::create(ov::element::i32, ov::Shape{}, {24});
    new_ops.insert(new_ops.end(), {mask_const, shift12, shift24});

    std::vector<std::vector<ov::Output<ov::Node>>> contributions(num_clean_limbs);
    for (size_t k = 0; k < num_mult_limbs; ++k) {
        auto limb_value = static_cast<int32_t>((multiplier >> (LIMB_BITS * static_cast<int64_t>(k))) & LIMB_MASK);
        auto limb_const = v0::Constant::create(ov::element::i32, ov::Shape{}, {limb_value});
        auto raw = std::make_shared<v1::Multiply>(token_i32, limb_const);
        new_ops.insert(new_ops.end(), {limb_const, raw});

        auto low = std::make_shared<v13::BitwiseAnd>(raw, mask_const);
        new_ops.push_back(low);
        contributions[k].push_back(low);

        auto shr_mid = std::make_shared<v15::BitwiseRightShift>(raw, shift12);
        auto mid = std::make_shared<v13::BitwiseAnd>(shr_mid, mask_const);
        new_ops.insert(new_ops.end(), {shr_mid, mid});
        contributions[k + 1].push_back(mid);

        // raw < 2**30 (token <= 18 bits), so raw >> 24 already fits in a limb.
        auto high = std::make_shared<v15::BitwiseRightShift>(raw, shift24);
        new_ops.push_back(high);
        contributions[k + 2].push_back(high);
    }

    std::vector<ov::Output<ov::Node>> clean(num_clean_limbs);
    ov::Output<ov::Node> carry;
    bool has_carry = false;
    for (size_t j = 0; j < num_clean_limbs; ++j) {
        ov::Output<ov::Node> total;
        bool has_total = false;
        for (auto& term : contributions[j]) {
            if (!has_total) {
                total = term;
                has_total = true;
            } else {
                auto add = std::make_shared<v1::Add>(total, term);
                new_ops.push_back(add);
                total = add;
            }
        }
        if (has_carry) {
            if (has_total) {
                auto add = std::make_shared<v1::Add>(total, carry);
                new_ops.push_back(add);
                total = add;
            } else {
                total = carry;
            }
            has_total = true;
        }
        if (!has_total) {
            total = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
            new_ops.push_back(total.get_node_shared_ptr());
        }

        auto clean_j = std::make_shared<v13::BitwiseAnd>(total, mask_const);
        new_ops.push_back(clean_j);
        clean[j] = clean_j;

        if (j + 1 < num_clean_limbs) {
            auto shr = std::make_shared<v15::BitwiseRightShift>(total, shift12);
            new_ops.push_back(shr);
            carry = shr;
            has_carry = true;
        }
    }
    return clean;
}

}  // namespace

ov::intel_cpu::NgramHashDecomposition::NgramHashDecomposition() {
    MATCHER_SCOPE(NgramHashDecomposition);
    auto add_m = ov::pass::pattern::wrap_type<v1::Add>();

    // Per-call cache so bigram/trigram groups sharing the same (token, multiplier) leaf don't
    // recompute identical limb decompositions (the leaves are literally the same shifted-token
    // tensor fed into separate Multiply nodes per group).
    using CacheKey = std::tuple<ov::Node*, int, int64_t>;
    auto limb_cache = std::make_shared<std::map<CacheKey, std::vector<ov::Output<ov::Node>>>>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto root_add = ov::as_type_ptr<v1::Add>(m.get_match_root());
        if (!root_add) {
            return false;
        }

        std::shared_ptr<v0::Constant> offsets_const;
        std::shared_ptr<v1::FloorMod> floor_mod;
        for (size_t order = 0; order < 2; ++order) {
            auto maybe_const = ov::as_type_ptr<v0::Constant>(root_add->input_value(1 - order).get_node_shared_ptr());
            auto maybe_floor_mod = ov::as_type_ptr<v1::FloorMod>(root_add->input_value(order).get_node_shared_ptr());
            if (maybe_const && maybe_floor_mod) {
                offsets_const = maybe_const;
                floor_mod = maybe_floor_mod;
                break;
            }
        }
        if (!offsets_const || !floor_mod) {
            return false;
        }

        auto moduli_const = ov::as_type_ptr<v0::Constant>(floor_mod->input_value(1).get_node_shared_ptr());
        if (!moduli_const) {
            return false;
        }
        auto moduli = moduli_const->cast_vector<int64_t>();
        if (moduli.empty() || std::any_of(moduli.begin(), moduli.end(), [](int64_t v) {
                return v <= 0;
            })) {
            return false;
        }
        const int64_t max_modulus = *std::max_element(moduli.begin(), moduli.end());
        // soundness bound for the 6-bit limb reduction below: 63 * p must fit int32.
        if (63 * max_modulus >= (int64_t{1} << 31)) {
            return false;
        }
        const size_t heads = moduli.size();

        auto offsets = offsets_const->cast_vector<int64_t>();
        if (offsets.size() != heads) {
            return false;
        }

        auto unsqueeze_node = floor_mod->input_value(0).get_node_shared_ptr();
        ov::NodeVector matched_nodes{root_add, offsets_const, floor_mod, moduli_const};
        // CommonOptimizations canonicalizes Unsqueeze(-1) into an equivalent Reshape; accept both.
        ov::Output<ov::Node> pre_unsqueeze;
        if (auto unsqueeze = ov::as_type_ptr<v0::Unsqueeze>(unsqueeze_node)) {
            pre_unsqueeze = unsqueeze->input_value(0);
            matched_nodes.push_back(unsqueeze);
        } else if (auto reshape = ov::as_type_ptr<v1::Reshape>(unsqueeze_node)) {
            pre_unsqueeze = reshape->input_value(0);
            matched_nodes.push_back(reshape);
        } else {
            return false;
        }

        std::vector<ov::Output<ov::Node>> dynamic_operands;
        std::vector<int64_t> multiplier_values;

        std::function<bool(const ov::Output<ov::Node>&)> collect = [&](const ov::Output<ov::Node>& out) -> bool {
            auto node = out.get_node_shared_ptr();
            if (auto xor_node = ov::as_type_ptr<v13::BitwiseXor>(node)) {
                matched_nodes.push_back(xor_node);
                return collect(xor_node->input_value(0)) && collect(xor_node->input_value(1));
            }
            if (auto mul_node = ov::as_type_ptr<v1::Multiply>(node)) {
                auto lhs = mul_node->input_value(0);
                auto rhs = mul_node->input_value(1);
                auto lhs_const = ov::as_type_ptr<v0::Constant>(lhs.get_node_shared_ptr());
                auto rhs_const = ov::as_type_ptr<v0::Constant>(rhs.get_node_shared_ptr());
                if (static_cast<bool>(lhs_const) == static_cast<bool>(rhs_const)) {
                    return false;  // need exactly one Constant side
                }
                auto dynamic_operand = lhs_const ? rhs : lhs;
                auto mult_const = lhs_const ? lhs_const : rhs_const;
                // CPU's ConvertPrecision unconditionally downcasts i64 to i32 for every op (not just
                // ones with out-of-i32-range constants), silently truncating the n-gram hash's
                // intermediate products: only i64 is actually broken here, so only engage for it.
                if (dynamic_operand.get_element_type() != ov::element::i64) {
                    return false;
                }
                auto mult_values = mult_const->cast_vector<int64_t>();
                if (mult_values.size() != 1 || mult_values[0] < 0) {
                    return false;  // scalar per-position multiplier only; negative not supported
                }
                matched_nodes.push_back(mul_node);
                matched_nodes.push_back(mult_const);
                dynamic_operands.push_back(dynamic_operand);
                multiplier_values.push_back(mult_values[0]);
                return true;
            }
            return false;
        };
        if (!collect(pre_unsqueeze) || dynamic_operands.size() < 2) {
            return false;
        }

        size_t num_mult_limbs = 1;
        for (auto v : multiplier_values) {
            num_mult_limbs = std::max<size_t>(num_mult_limbs, static_cast<size_t>((bit_length(v) + LIMB_BITS - 1) / LIMB_BITS));
        }
        const size_t num_clean_limbs = num_mult_limbs + SPREAD_LIMBS;
        if (static_cast<int64_t>(2 * num_clean_limbs) * max_modulus >= (int64_t{1} << 31)) {
            return false;
        }

        ov::NodeVector new_ops;
        std::vector<std::vector<ov::Output<ov::Node>>> operand_clean_limbs;
        for (size_t i = 0; i < dynamic_operands.size(); ++i) {
            CacheKey key{dynamic_operands[i].get_node(), dynamic_operands[i].get_index(), multiplier_values[i]};
            auto cached = limb_cache->find(key);
            if (cached != limb_cache->end() && cached->second.size() == num_clean_limbs) {
                operand_clean_limbs.push_back(cached->second);
                continue;
            }
            auto limbs = build_clean_limbs(dynamic_operands[i], multiplier_values[i], num_mult_limbs, new_ops);
            (*limb_cache)[key] = limbs;
            operand_clean_limbs.push_back(std::move(limbs));
        }

        std::vector<ov::Output<ov::Node>> mixed_limbs(num_clean_limbs);
        for (size_t j = 0; j < num_clean_limbs; ++j) {
            ov::Output<ov::Node> acc = operand_clean_limbs[0][j];
            for (size_t op_idx = 1; op_idx < operand_clean_limbs.size(); ++op_idx) {
                auto xor_node = std::make_shared<v13::BitwiseXor>(acc, operand_clean_limbs[op_idx][j]);
                new_ops.push_back(xor_node);
                acc = xor_node;
            }
            mixed_limbs[j] = acc;
        }

        // Big integer in 12-bit limbs modulo several primes at once, as
        // `sum_k limb6_k * (2**(6k) mod p) mod p` in int32, with all 6-bit limbs stacked on one extra
        // axis so the per-term reduction is a single subgraph.
        // 6-bit limbs keep every product below 2**31 given the `63 * p < 2**31` bound above, and the sum
        // of the 2 * num_clean_limbs reduced terms stays below 2**31 (checked when matching).
        std::vector<int32_t> moduli_i32(moduli.begin(), moduli.end());
        std::vector<float> moduli_f32(moduli.begin(), moduli.end());
        auto moduli_const_i32 = v0::Constant::create(ov::element::i32, ov::Shape{heads}, moduli_i32);
        auto moduli_const_f32 = v0::Constant::create(ov::element::f32, ov::Shape{heads}, moduli_f32);
        auto mask63 = v0::Constant::create(ov::element::i32, ov::Shape{}, {63});
        auto shift6 = v0::Constant::create(ov::element::i32, ov::Shape{}, {6});
        auto last_axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto limb_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {-2});
        new_ops.insert(new_ops.end(), {moduli_const_i32, moduli_const_f32, mask63, shift6, last_axis, limb_axis});

        ov::OutputVector limb_columns;
        for (auto& limb : mixed_limbs) {
            auto column = std::make_shared<v0::Unsqueeze>(limb, last_axis);
            new_ops.push_back(column);
            limb_columns.push_back(column);
        }
        auto limbs12 = std::make_shared<v0::Concat>(limb_columns, -1);
        auto low6 = std::make_shared<v13::BitwiseAnd>(limbs12, mask63);
        // limb < 2**12, so limb >> 6 already fits 6 bits: no extra mask needed.
        auto high6 = std::make_shared<v15::BitwiseRightShift>(limbs12, shift6);
        auto limbs6 = std::make_shared<v0::Concat>(ov::OutputVector{low6, high6}, -1);
        auto limbs6_column = std::make_shared<v0::Unsqueeze>(limbs6, last_axis);
        new_ops.insert(new_ops.end(), {limbs12, low6, high6, limbs6, limbs6_column});

        // limbs6 is [low_0 .. low_{L-1}, high_0 .. high_{L-1}]: low_j is 6-bit limb 2j, high_j is limb 2j+1.
        std::vector<int32_t> weight_vals(2 * num_clean_limbs * heads);
        for (size_t i = 0; i < 2 * num_clean_limbs; ++i) {
            const auto k = static_cast<int64_t>(i < num_clean_limbs ? 2 * i : 2 * (i - num_clean_limbs) + 1);
            for (size_t h = 0; h < heads; ++h) {
                weight_vals[i * heads + h] = static_cast<int32_t>(pow_mod(2, 6 * k, moduli[h]));
            }
        }
        auto weights = v0::Constant::create(ov::element::i32, ov::Shape{2 * num_clean_limbs, heads}, weight_vals);
        auto weighted = std::make_shared<v1::Multiply>(limbs6_column, weights);
        new_ops.insert(new_ops.end(), {weights, weighted});

        auto terms = make_mod_exact(weighted, moduli_const_i32, moduli_const_f32, new_ops);
        // CPU ReduceSum accumulates i32 through f32 (inexact above 2**24), so sum with exact i32 Adds.
        auto split = std::make_shared<v1::Split>(terms, limb_axis, 2 * num_clean_limbs);
        new_ops.push_back(split);
        ov::Output<ov::Node> sum = split->output(0);
        for (size_t i = 1; i < 2 * num_clean_limbs; ++i) {
            auto add = std::make_shared<v1::Add>(sum, split->output(i));
            new_ops.push_back(add);
            sum = add;
        }
        auto total = std::make_shared<v0::Squeeze>(sum, limb_axis);
        new_ops.push_back(total);
        auto reduced = make_mod_exact(total, moduli_const_i32, moduli_const_f32, new_ops);

        auto reduced_as_orig_type = std::make_shared<v0::Convert>(reduced, offsets_const->get_element_type());
        auto result = std::make_shared<v1::Add>(reduced_as_orig_type, offsets_const);
        new_ops.insert(new_ops.end(), {reduced_as_orig_type, result});

        result->set_friendly_name(root_add->get_friendly_name());
        ov::copy_runtime_info(matched_nodes, new_ops);
        ov::replace_node(root_add, result);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(add_m, matcher_name);
    register_matcher(m, callback);
}
