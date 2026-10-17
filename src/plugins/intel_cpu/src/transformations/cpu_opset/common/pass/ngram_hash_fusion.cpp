// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngram_hash_fusion.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/bitwise_xor.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/cpu_opset/common/op/ngram_hash.hpp"

using namespace ov::op;

ov::intel_cpu::NgramHashFusion::NgramHashFusion() {
    MATCHER_SCOPE(NgramHashFusion);
    auto add_m = ov::pass::pattern::wrap_type<v1::Add>();

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
        auto offsets = offsets_const->cast_vector<int64_t>();
        if (offsets.size() != moduli.size()) {
            return false;
        }
        // ConvertPrecision still downcasts the fused node's i64 output to i32, so every id must fit.
        for (size_t h = 0; h < moduli.size(); ++h) {
            if (offsets[h] < 0 || offsets[h] + moduli[h] - 1 > std::numeric_limits<int32_t>::max()) {
                return false;
            }
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

        ov::OutputVector tokens;
        std::vector<int64_t> multipliers;
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
                auto token = lhs_const ? rhs : lhs;
                auto mult_const = lhs_const ? lhs_const : rhs_const;
                // CPU's ConvertPrecision downcasts every i64 op to i32, silently wrapping the hash's
                // intermediate products: only i64 is actually broken here, so only engage for it.
                if (token.get_element_type() != ov::element::i64) {
                    return false;
                }
                auto mult_values = mult_const->cast_vector<int64_t>();
                if (mult_values.size() != 1) {
                    return false;  // scalar per-position multiplier only
                }
                matched_nodes.push_back(mul_node);
                matched_nodes.push_back(mult_const);
                tokens.push_back(token);
                multipliers.push_back(mult_values[0]);
                return true;
            }
            return false;
        };
        if (!collect(pre_unsqueeze) || tokens.size() < 2) {
            return false;
        }

        auto token_shape = tokens[0].get_partial_shape();
        for (const auto& token : tokens) {
            if (!ov::PartialShape::merge_into(token_shape, token.get_partial_shape())) {
                return false;
            }
        }

        auto fused = std::make_shared<ov::intel_cpu::NgramHashNode>(tokens, multipliers, moduli, offsets);
        if (fused->get_output_element_type(0) != root_add->get_output_element_type(0) ||
            !fused->get_output_partial_shape(0).compatible(root_add->get_output_partial_shape(0))) {
            return false;
        }

        fused->set_friendly_name(root_add->get_friendly_name());
        ov::copy_runtime_info(matched_nodes, fused);
        ov::replace_node(root_add, fused);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(add_m, matcher_name);
    register_matcher(m, callback);
}
