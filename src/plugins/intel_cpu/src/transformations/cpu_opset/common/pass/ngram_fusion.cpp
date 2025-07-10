// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngram_fusion.hpp"

#include <cstddef>
#include <memory>
#include <openvino/core/dimension.hpp>
#include <openvino/core/graph_util.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/symbol.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "transformations/cpu_opset/common/op/ngram.hpp"

using namespace ov::pass::pattern;
ov::intel_cpu::NgramFusion::NgramFusion() {
    MATCHER_SCOPE(NgramFusion);
    auto concat_matches = [](const ov::Output<ov::Node>& output) -> bool {
        if (auto concat = ov::as_type_ptr<ov::op::v0::Concat>(output.get_node_shared_ptr())) {
            return ov::pass::pattern::rank_equals(2)(output) && concat->get_axis() == 1;
        }
        return false;
    };
    auto concat_m = ov::pass::pattern::wrap_type<ov::op::v0::Concat>(concat_matches);

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto concat = m.get_match_root();
        const auto& inputs = concat->input_values();

        const size_t k = inputs.size();
        const size_t as_is_idx = k % 2 == 0 ? (k - 1) / 2 : k / 2;
        // Ngram must contain only one non-select branch (that propagates original tensor as is)
        for (size_t i = 0; i < inputs.size(); ++i) {
            const bool is_select = ov::is_type<ov::op::v1::Select>(inputs[i].get_node());
            if ((i == as_is_idx && is_select) || (i != as_is_idx && !is_select)) {
                return false;
            }
        }

        auto check_bias = [](const PatternValueMap& pattern_map,
                             const std::shared_ptr<ov::Node>& matched_constant_to_check,
                             const size_t expected_bias) {
            auto out_it = pattern_map.find(matched_constant_to_check);
            if (expected_bias == 0) {
                return out_it == pattern_map.end();
            }
            if (out_it == pattern_map.end()) {
                return false;
            }
            const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(out_it->second.get_node_shared_ptr());
            return constant != nullptr && ov::op::util::constantIsEqualTo(constant, expected_bias);
        };

        auto tokens_match = [](const ov::Output<ov::Node>& output) -> bool {
            return ov::pass::pattern::rank_equals(2)(output) &&
                   ov::pass::pattern::type_matches(ov::element::f32)(output);
        };
        auto idces_match = [](const ov::Output<ov::Node>& output) -> bool {
            return ov::pass::pattern::rank_equals(2)(output) &&
                   ov::pass::pattern::type_matches(ov::element::i32)(output);
        };
        auto as_is_cropped_shape_match = [](const ov::Output<ov::Node>& output) -> bool {
            const auto& symbols = output.get_tensor().get_value_symbol();
            return ov::pass::pattern::rank_equals(1)(output) && !symbols.empty() && symbols[0] != nullptr;
        };

        std::shared_ptr<Symbol> cropped_shape_symbol = nullptr;

        ov::Output<ov::Node> tokens;
        // "as_is" input validation
        {
            auto tokens_m = any_input(tokens_match);
            auto padded_tokens_inputs =
                as_is_idx == 0
                    ? ov::OutputVector{tokens_m, wrap_type<ov::op::v0::Constant>()}
                    : ov::OutputVector{wrap_type<ov::op::v0::Constant>(), tokens_m, wrap_type<ov::op::v0::Constant>()};
            auto padded_tokens_m = wrap_type<ov::op::v0::Concat>(padded_tokens_inputs);
            auto cropped_shape_m = any_input(as_is_cropped_shape_match);
            auto ss_bias = wrap_type<ov::op::v0::Constant>();
            auto ss_biased_shape_m =
                as_is_idx == 0 ? cropped_shape_m : wrap_type<ov::op::v1::Add>({cropped_shape_m, ss_bias});
            auto cropped_tokens_m = wrap_type<ov::op::v1::StridedSlice>({padded_tokens_m,
                                                                         wrap_type<ov::op::v0::Constant>(),
                                                                         ss_biased_shape_m,
                                                                         wrap_type<ov::op::v0::Constant>()});
            Matcher matcher(cropped_tokens_m);

            if (!matcher.match(inputs[as_is_idx])) {
                return false;
            }

            const auto& pattern_map = matcher.get_pattern_value_map();
            tokens = pattern_map.at(tokens_m);
            const auto& concat_shape = concat->get_output_partial_shape(0);
            const auto& tokens_shape = tokens.get_partial_shape();

            // To confirm that a subgraph can be replaced with NgramNode we only need to
            // 1. Check Add's constant to make sure that data values have a right bias in result tensor
            // 2. Check subgraph input and output shapes to make sure that all rest constants in the subgraph are
            // correct
            if (!check_bias(pattern_map, ss_bias, as_is_idx) || concat_shape.rank() != tokens_shape.rank() ||
                tokens_shape[1] * k != concat_shape[1]) {
                return false;
            }
            // save symbol of cropped_shape and check it against first dimension of tokens shape
            cropped_shape_symbol = pattern_map.at(cropped_shape_m).get_tensor().get_value_symbol()[0];
            if (!symbol::are_equal(tokens_shape[0].get_symbol(), cropped_shape_symbol)) {
                return false;
            }
        }

        auto cropped_shape_symbol_match = [cropped_shape_symbol](const ov::Output<ov::Node>& output) -> bool {
            const auto& symbols = output.get_tensor().get_value_symbol();
            return ov::pass::pattern::rank_equals(1)(output) && !symbols.empty() &&
                   ov::symbol::are_equal(symbols[0], cropped_shape_symbol);
        };

        auto tokens_symbol_match = [tokens_match, cropped_shape_symbol](const ov::Output<ov::Node>& output) -> bool {
            return tokens_match(output) &&
                   symbol::are_equal(output.get_partial_shape()[0].get_symbol(), cropped_shape_symbol);
        };

        ov::Output<ov::Node> indices;
        // select branches validation
        {
            auto tokens_m = any_input(tokens_symbol_match);
            auto cropped_shape_m = any_input(cropped_shape_symbol_match);
            auto idces_m = any_input(idces_match);
            auto idces_concat_inputs =
                as_is_idx == 0
                    ? ov::OutputVector{idces_m, wrap_type<ov::op::v0::Constant>()}
                    : ov::OutputVector{wrap_type<ov::op::v0::Constant>(), idces_m, wrap_type<ov::op::v0::Constant>()};
            auto idces_concat_m = wrap_type<ov::op::v0::Concat>(idces_concat_inputs);

            // Concat can be replaced by symbolic optimizations as it can find alternative source for this op.
            const auto concat_shape_with_cropped_symbol_m =
                [&cropped_shape_symbol](const ov::Output<ov::Node>& output) -> bool {
                const auto& symbols = output.get_tensor().get_value_symbol();
                return ov::pass::pattern::rank_equals(1)(output) && !symbols.empty() &&
                       symbols[0] == cropped_shape_symbol;
            };

            // left equal branch
            auto crop_left_bias_m = wrap_type<ov::op::v0::Constant>();
            auto crop_left_cropped_shape_m = std::make_shared<ov::pass::pattern::op::Or>(
                ov::OutputVector{wrap_type<ov::op::v1::Add>({cropped_shape_m, crop_left_bias_m}), cropped_shape_m});
            auto idxes_crop_left_concat_m = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{
                wrap_type<ov::op::v0::Concat>({crop_left_cropped_shape_m, wrap_type<ov::op::v0::Constant>()}),
                any_input(concat_shape_with_cropped_symbol_m)});
            auto idxes_crop_left_m = wrap_type<ov::op::v1::StridedSlice>({idces_concat_m,
                                                                          wrap_type<ov::op::v0::Constant>(),
                                                                          idxes_crop_left_concat_m,
                                                                          wrap_type<ov::op::v0::Constant>()});

            // right equal branch
            auto crop_right_bias_m = wrap_type<ov::op::v0::Constant>();
            auto crop_right_cropped_shape_m = std::make_shared<ov::pass::pattern::op::Or>(
                ov::OutputVector{wrap_type<ov::op::v1::Add>({cropped_shape_m, crop_right_bias_m}), cropped_shape_m});
            auto idxes_crop_right_concat_m = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{
                wrap_type<ov::op::v0::Concat>({crop_right_cropped_shape_m, wrap_type<ov::op::v0::Constant>()}),
                any_input(concat_shape_with_cropped_symbol_m)});
            auto idxes_crop_right_m = wrap_type<ov::op::v1::StridedSlice>({idces_concat_m,
                                                                           wrap_type<ov::op::v0::Constant>(),
                                                                           idxes_crop_right_concat_m,
                                                                           wrap_type<ov::op::v0::Constant>()});

            auto equal_m = wrap_type<ov::op::v1::Equal>({idxes_crop_left_m, idxes_crop_right_m});
            auto condition_m = wrap_type<ov::op::v1::Reshape>({equal_m, any_input()});

            // then branch
            auto padded_tokens_inputs =
                as_is_idx == 0
                    ? ov::OutputVector{tokens_m, wrap_type<ov::op::v0::Constant>()}
                    : ov::OutputVector{wrap_type<ov::op::v0::Constant>(), tokens_m, wrap_type<ov::op::v0::Constant>()};
            auto padded_tokens_m = wrap_type<ov::op::v0::Concat>(padded_tokens_inputs);

            auto then_cropped_shape_bias_m = wrap_type<ov::op::v0::Constant>();
            auto then_cropped_shape_m = std::make_shared<ov::pass::pattern::op::Or>(
                ov::OutputVector{wrap_type<ov::op::v1::Add>({cropped_shape_m, then_cropped_shape_bias_m}),
                                 cropped_shape_m});
            auto then_m = wrap_type<ov::op::v1::StridedSlice>({padded_tokens_m,
                                                               wrap_type<ov::op::v0::Constant>(),
                                                               then_cropped_shape_m,
                                                               wrap_type<ov::op::v0::Constant>()});

            // else branch
            auto else_target_shape_concat_m = any_input(concat_shape_with_cropped_symbol_m);
            auto else_m = wrap_type<ov::op::v1::Broadcast>(
                {wrap_type<ov::op::v0::Constant>(), else_target_shape_concat_m, wrap_type<ov::op::v0::Constant>()});
            auto select_m = wrap_type<ov::op::v1::Select>({condition_m, then_m, else_m});
            Matcher select_matcher(select_m);

            for (size_t i = 0; i < inputs.size(); ++i) {
                if (i == as_is_idx) {
                    continue;
                }
                if (!select_matcher.match(inputs[i])) {
                    return false;
                }

                const auto& pattern_map = select_matcher.get_pattern_value_map();
                const auto& cur_tokens_input = pattern_map.at(tokens_m);
                const auto& cur_indices_input = pattern_map.at(idces_m);
                if (indices.get_node() == nullptr) {
                    indices = cur_indices_input;
                }

                // To confirm that a subgraph can be replaced with NgramNode we need to
                // 1. Check that "tokens" input is equal to the input that was matched earlier
                // 2. Check that "indices" input for all Select branches is the same
                // 3. Check Add's constants to make sure that data values have a right bias in result tensor
                const bool validate_eq_biases = (check_bias(pattern_map, crop_left_bias_m, i) &&
                                                 check_bias(pattern_map, crop_right_bias_m, as_is_idx)) ||
                                                (check_bias(pattern_map, crop_right_bias_m, i) &&
                                                 check_bias(pattern_map, crop_left_bias_m, as_is_idx));
                if (cur_tokens_input != tokens || cur_indices_input != indices || !validate_eq_biases ||
                    !check_bias(pattern_map, then_cropped_shape_bias_m, i)) {
                    return false;
                }
            }
        }

        const auto ngram = std::make_shared<ov::intel_cpu::NgramNode>(tokens, indices, k);
        ngram->set_friendly_name(concat->get_friendly_name());
        ov::copy_runtime_info(concat, ngram);
        ov::replace_node(concat, ngram);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(concat_m, matcher_name);
    this->register_matcher(m, callback);
}
