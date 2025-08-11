// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/move_fake_convert_up_through_kv_cache_concat.hpp"

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/fake_convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::pass::low_precision {
MoveFakeConvertUpThroughKVCacheConcat::MoveFakeConvertUpThroughKVCacheConcat() {
    MATCHER_SCOPE(MoveFakeConvertUpThroughKVCacheConcat);
    using namespace ov::pass::pattern;

    auto match_kv_cache = []() {
        auto past_k = wrap_type<ov::op::v6::ReadValue>({});
        auto convert_kv = optional<ov::op::v0::Convert>({past_k});
        auto gather_input_k = wrap_type<ov::op::v8::Gather>({convert_kv, any_input(), any_input()});
        auto cache_input = wrap_type<ov::op::v0::Parameter>() | gather_input_k;
        auto cur_kv = any_input();
        auto kv_concat = wrap_type<ov::op::v0::Concat>({cache_input, cur_kv});

        // Note: SDPA can be quantized only with scalar scale and shift.
        auto fc_scale = wrap_type<ov::op::v0::Constant>(rank_equals(0));
        auto fc_shift = wrap_type<ov::op::v0::Constant>(rank_equals(0));
        auto fake_convert = wrap_type<ov::op::v13::FakeConvert>({kv_concat, fc_scale}) |
                            wrap_type<ov::op::v13::FakeConvert>({kv_concat, fc_scale, fc_shift});

        auto [kv, reshape_kv, unsqueeze_kv, computed_bcst_kv, multiply_kv, computed_bcst3_kv] =
            ov::op::util::match_multi_query_bcst(fake_convert);
        return std::make_tuple(kv_concat, fake_convert, kv);
    };

    auto k_cache_result = match_kv_cache();
    auto k_concat = std::get<0>(k_cache_result);
    auto k_fc = std::get<1>(k_cache_result);
    auto key = std::get<2>(k_cache_result);

    auto v_cache_result = match_kv_cache();
    auto v_concat = std::get<0>(v_cache_result);
    auto v_fc = std::get<1>(v_cache_result);
    auto value = std::get<2>(v_cache_result);

#define ANY any_input()
    auto sdpa_m = wrap_type<ov::op::v13::ScaledDotProductAttention>({ANY, key, value}) |
                  wrap_type<ov::op::v13::ScaledDotProductAttention>({ANY, key, value, ANY}) |
                  wrap_type<ov::op::v13::ScaledDotProductAttention>({ANY, key, value, ANY, ANY});
#undef ANY

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        bool res = false;
        const auto& pattern_map = m.get_pattern_value_map();
        for (const auto& pattern : {std::make_pair(k_concat, k_fc), std::make_pair(v_concat, v_fc)}) {
            const auto& [concat_m, fake_convert_m] = pattern;
            const auto concat = pattern_map.at(concat_m).get_node_shared_ptr();
            const auto concat_consumers = concat->get_users();
            if (std::any_of(concat_consumers.begin(),
                            concat_consumers.end(),
                            [](const std::shared_ptr<ov::Node>& consumer) {
                                return !ov::is_type_any_of<ov::op::v13::FakeConvert,
                                                           ov::op::v0::Result,
                                                           ov::op::v6::Assign,
                                                           ov::op::v3::ShapeOf>(consumer);
                            })) {
                continue;
            }

            const auto fake_convert = pattern_map.at(fake_convert_m).get_node_shared_ptr();
            ov::replace_output_update_name(fake_convert->output(0), fake_convert->input_value(0));

            auto form_new_fc_inputs = [fake_convert](const ov::Output<ov::Node>& new_input) {
                auto new_inputs = fake_convert->input_values();
                new_inputs[0] = new_input;
                return new_inputs;
            };

            auto fc_cache = fake_convert->clone_with_new_inputs(form_new_fc_inputs(concat->input_value(0)));
            auto fc_kv = fake_convert->clone_with_new_inputs(form_new_fc_inputs(concat->input_value(1)));
            fc_cache->set_friendly_name(fake_convert->get_friendly_name() + "_1");
            fc_kv->set_friendly_name(fake_convert->get_friendly_name() + "_2");
            ov::copy_runtime_info(fake_convert, {fc_cache, fc_kv});

            auto new_concat = concat->clone_with_new_inputs({fc_cache->output(0), fc_kv->output(0)});
            new_concat->set_friendly_name(concat->get_friendly_name());
            ov::copy_runtime_info(concat, new_concat);
            ov::replace_node(concat, new_concat);
            res = true;
        }
        return res;
    };

    auto m = std::make_shared<Matcher>(sdpa_m, matcher_name);
    register_matcher(m, callback);
}
}  // namespace ov::pass::low_precision
