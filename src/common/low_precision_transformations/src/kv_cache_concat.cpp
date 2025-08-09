// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/kv_cache_concat.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "itt.hpp"
#include "low_precision/network_helper.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/fake_convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::pass::low_precision {
using namespace ov;

KVCacheConcat::KVCacheConcat(const std::shared_ptr<Model>& model) {
    MATCHER_SCOPE(KVCacheConcat);
    using namespace ov::pass::pattern;
    using namespace ov::op;

    auto read_value_m = wrap_type<v6::ReadValue>({});
    auto gather_m = wrap_type<v8::Gather>({read_value_m, any_input(), any_input()});

    auto match_downconvert_subgraph = [](const Output<Node>& input) {
        auto scale = wrap_type<v0::Constant>(rank_equals(0));
        auto mul = wrap_type<v1::Multiply>({input, scale});

        auto shift = wrap_type<v0::Constant>(rank_equals(0));
        auto sub = optional<v1::Subtract>({mul, shift});

        auto clamp = wrap_type<v0::Clamp>({sub});
        auto downconvert = wrap_type<v0::Convert>({clamp}, type_matches_any({element::f8e4m3, element::f8e5m2}));
        return std::make_tuple(scale, shift, downconvert);
    };

    auto cache_patterns = match_downconvert_subgraph(gather_m);
    auto down_scale_cache = std::get<0>(cache_patterns);
    auto down_shift_cache = std::get<1>(cache_patterns);
    auto downconvert_cache = std::get<2>(cache_patterns);

    auto kv_patterns = match_downconvert_subgraph(any_input());
    auto down_scale_kv = std::get<0>(kv_patterns);
    auto down_shift_kv = std::get<1>(kv_patterns);
    auto downconvert_kv = std::get<2>(kv_patterns);
    auto kv_concat = wrap_type<v0::Concat>({downconvert_cache, downconvert_kv});

    auto upconvert = wrap_type<v0::Convert>({kv_concat});
    auto up_shift = wrap_type<v0::Constant>();
    auto up_sub = optional<v1::Subtract>({upconvert, up_shift});
    auto up_scale = wrap_type<v0::Constant>();
    auto up_mul = wrap_type<v1::Multiply>({up_sub, up_scale});

    auto kv_assign = wrap_type<v6::Assign>({up_mul});

    graph_rewrite_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        const auto& pattern_map = m.get_pattern_value_map();
        auto check_dequantization_pairs = [&pattern_map](const std::shared_ptr<Node>& const_cache,
                                                         const std::shared_ptr<Node>& const_kv) {
            if (pattern_map.count(const_cache) != pattern_map.count(const_kv)) {
                return false;
            }
            if (pattern_map.count(const_cache) == 0) {
                return true;
            }
            const auto cache_node = pattern_map.at(const_cache).get_node_shared_ptr();
            const auto kv_node = pattern_map.at(const_kv).get_node_shared_ptr();
            const auto equal_node = ov::as_type_ptr<v0::Constant>(fold<v1::Equal>(cache_node, kv_node));
            OPENVINO_ASSERT(equal_node != nullptr, "Downconvert scaleshift must be constant.");
            const auto equal_res = equal_node->get_vector<bool>();
            // Note: the optimization can be applied only if quantization scales and shifts on both
            // concat branches are equal.
            return std::all_of(equal_res.begin(), equal_res.end(), [](bool x) {
                return x;
            });
        };

        if (!check_dequantization_pairs(down_scale_cache, down_scale_kv) ||
            !check_dequantization_pairs(down_shift_cache, down_shift_kv)) {
            return false;
        }
        const auto assign = ov::as_type_ptr<v6::Assign>(pattern_map.at(kv_assign).get_node_shared_ptr());
        const auto read_value = ov::as_type_ptr<v6::ReadValue>(pattern_map.at(read_value_m).get_node_shared_ptr());
        OPENVINO_ASSERT(assign != nullptr && read_value != nullptr, "Assign node must be found in the pattern map.");
        if (assign->get_control_dependencies().size() != 1 || assign->get_control_dependencies()[0] != read_value) {
            return false;
        }

        // Note: this dependency must be removed before read value update.
        assign->remove_control_dependency(read_value);
        const auto original_var = read_value->get_variable();
        const auto var_info = original_var->get_info();
        const auto& down_convert_prc = pattern_map.at(downconvert_cache).get_element_type();
        // set new precision for original_var to update precision in new read value
        original_var->update({var_info.data_shape, down_convert_prc, var_info.variable_id});
        const auto read_value_convert = std::make_shared<v0::Convert>(read_value->input_value(0), down_convert_prc);
        const auto new_read_value = read_value->copy_with_new_inputs({read_value_convert});
        OPENVINO_ASSERT(replace_node_update_name(read_value, new_read_value), "Failed to update ReadValue node.");
        copy_runtime_info(read_value, read_value_convert);

        // Manual gather validation is called in order to propagate low precision through it
        const auto gather_node = pattern_map.at(gather_m).get_node_shared_ptr();
        gather_node->validate_and_infer_types();

        auto build_reverse_fake_convert = [&](ov::Output<ov::Node> input) {
            auto clone_constant = [&](const std::shared_ptr<Node>& node) {
                return pattern_map.at(node).get_node_shared_ptr()->clone_with_new_inputs({});
            };

            auto upconvert_node = pattern_map.at(upconvert).get_node_shared_ptr()->clone_with_new_inputs({input});
            std::shared_ptr<ov::Node> up_sub_node = upconvert_node;
            if (pattern_map.count(up_shift)) {
                up_sub_node = std::make_shared<v1::Subtract>(upconvert_node, clone_constant(up_shift));
            }
            auto up_mul_node = std::make_shared<v1::Multiply>(up_sub_node, clone_constant(up_scale));
            auto down_mul_node = std::make_shared<v1::Multiply>(up_mul_node, clone_constant(down_scale_cache));
            std::shared_ptr<ov::Node> down_sub_node = down_mul_node;
            if (pattern_map.count(down_shift_cache)) {
                down_sub_node = std::make_shared<v1::Subtract>(down_mul_node, clone_constant(down_shift_cache));
            }
            auto downconvert_node =
                pattern_map.at(downconvert_cache).get_node_shared_ptr()->clone_with_new_inputs({down_sub_node});
            copy_runtime_info(
                input.get_node_shared_ptr(),
                {upconvert_node, up_sub_node, up_mul_node, down_mul_node, down_sub_node, downconvert_node});
            return downconvert_node;
        };

        // Note: we create "reverse" fake convert here (low precision on input-output, high precision (e.g. f32) inside)
        // in order to keep quantization parameters on concat inputs and output
        auto reverse_fc_before_concat = build_reverse_fake_convert(gather_node);
        auto reverse_fc_after_concat = build_reverse_fake_convert(pattern_map.at(kv_concat));
        OPENVINO_ASSERT(replace_output_update_name(pattern_map.at(downconvert_cache), reverse_fc_before_concat),
                        "Failed to replace downconvert subgraph from kv cache branch.");

        // Assign is reconnected to the reverse fake convert output
        const auto new_assign = assign->copy_with_new_inputs({reverse_fc_after_concat});
        model->remove_sink(as_type_ptr<Sink>(assign));
        model->add_sinks({as_type_ptr<Sink>(new_assign)});

        OPENVINO_ASSERT(replace_node_update_name(assign, new_assign), "Failed to update Assign node.");
        new_assign->add_control_dependency(new_read_value);
        return true;
    };

    auto matcher = std::make_shared<Matcher>(kv_assign, matcher_name);

    register_matcher(matcher, callback);
}
}  // namespace ov::pass::low_precision
