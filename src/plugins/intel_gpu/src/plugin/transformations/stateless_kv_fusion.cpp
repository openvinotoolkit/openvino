// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "stateless_kv_fusion.hpp"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string_view>

#include "intel_gpu/op/stateless_kv.hpp"
#include "intel_gpu/op/sdpa.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

StatelessKVFusionMatcher::StatelessKVFusionMatcher() {
    using namespace ov::pass::pattern;
    using namespace ov::op;

    auto past = wrap_type<ov::op::v0::Parameter>();
    auto new_token_data = any_input();

    auto total_seqlen = wrap_type<ov::op::v0::Parameter>(shape_matches("[1]"));
    auto total_seqlen_cvt = wrap_type<ov::op::v0::Convert>({total_seqlen});
    auto total_seqlen_actual = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{total_seqlen, total_seqlen_cvt});
    auto seqlens_k = wrap_type<ov::op::v0::Parameter>(shape_matches("[1,1]"));
    auto seqlens_k_cvt = wrap_type<ov::op::v0::Convert>({seqlens_k});
    auto seqlens_k_actual = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{seqlens_k, seqlens_k_cvt});
    auto real_seqlens = wrap_type<ov::op::v1::Add>({seqlens_k_actual, 1});
    auto seqlens_1d = wrap_type<ov::op::v1::Reshape>({real_seqlens, 1});
    auto concat_kv_len = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{total_seqlen, total_seqlen_cvt, seqlens_1d});

    auto cur_seqlen_shapeof = wrap_type<ov::op::v3::ShapeOf>({any_input()});
    auto seqlen_dim = wrap_type<ov::op::v0::Constant>(shape_matches("[1]"));
    auto cur_seqlen = wrap_type<ov::op::v8::Gather>({cur_seqlen_shapeof, seqlen_dim, 0});
    auto cur_seqlen_neg = wrap_type<ov::op::v1::Multiply>({cur_seqlen, -1});
    auto cur_seqlen_neg_const = wrap_type<ov::op::v0::Constant>(shape_matches("[?]"));
    auto past_seqlen_add =
        wrap_type<ov::op::v1::Add>({concat_kv_len, std::make_shared<ov::pass::pattern::op::Or>(OutputVector{cur_seqlen_neg, cur_seqlen_neg_const})});
    auto past_seqlen_sub = wrap_type<ov::op::v1::Subtract>({concat_kv_len, cur_seqlen});

    auto range_cur = wrap_type<ov::op::v4::Range>({0, any_input(), 1});
    auto const_range_cur = wrap_type<ov::op::v0::Constant>(shape_matches("[?]"));
    auto pos_idx_base = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{range_cur, const_range_cur});
    auto past_seqlen_from_param = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{past_seqlen_add, past_seqlen_sub, any_input()});
    auto shifted_pos_idx = wrap_type<ov::op::v1::Add>({pos_idx_base, past_seqlen_from_param});
    auto pos_idx = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{shifted_pos_idx, any_input()});
    auto scatter_axis = wrap_type<ov::op::v0::Constant>(shape_matches("[1]"));
    auto scatter_update = wrap_type<ov::op::v3::ScatterUpdate>({past, pos_idx, new_token_data, scatter_axis});

    auto slice_axis = wrap_type<ov::op::v0::Constant>();
    auto past_seqlen_actual = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{past_seqlen_add, past_seqlen_sub, any_input()});
    auto slice = wrap_type<ov::op::v8::Slice>({past, 0, past_seqlen_actual, 1, slice_axis});
    auto concat = wrap_type<ov::op::v0::Concat>({slice, new_token_data});

    auto kv_actual = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{scatter_update, concat});
    auto result = wrap_type<ov::op::v0::Result>({kv_actual});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto result_node = ov::as_type_ptr<ov::op::v0::Result>(pattern_map.at(result).get_node_shared_ptr());
        const auto result_input = result_node->input(0);
        const auto kv_present_output = result_input.get_source_output();
        const auto past_output = pattern_map.at(past);
        const auto new_token_output = pattern_map.at(new_token_data);
        const auto new_token_shape = new_token_output.get_partial_shape();
        std::optional<ov::Input<ov::Node>> kv_to_sdpa_input;
        std::optional<ov::Input<ov::Node>> kv_sdpa_input;
        ov::Output<ov::Node> pos_idx_output;
        ov::Output<ov::Node> seqlen_output;
        ov::NodeVector node_infos;
        std::vector<ov::Input<ov::Node>> other_inputs;
        int64_t target_axis = 0;
        bool is_present_len = true;

        const bool is_slice_concat = pattern_map.count(concat) > 0;
        bool is_update_split = false;

        for (const auto& input : kv_present_output.get_target_inputs()) {
            if (input == result_input) {
                continue;
            }
            ov::Node* shapeof_node = ov::as_type<ov::op::v3::ShapeOf>(input.get_node());
            // first non shape-of will be treated as the input to SDPA
            if (!ov::as_type<ov::op::v3::ShapeOf>(input.get_node()) && !kv_to_sdpa_input.has_value()) {
                kv_to_sdpa_input.emplace(input);
            } else {
                other_inputs.push_back(input);
            }
        }
        if (!kv_to_sdpa_input.has_value()) {
            return false;
        }

        std::optional<int64_t> shapeof_axis;
        if (pattern_map.count(seqlen_dim) > 0) {
            auto seqlen_dim_node = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(seqlen_dim).get_node_shared_ptr());
            shapeof_axis = seqlen_dim_node->cast_vector<int64_t>()[0];
        }
        std::optional<int64_t> neg_cur_seqlen;
        if (pattern_map.count(cur_seqlen_neg_const) > 0) {
            auto cur_neg_node = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(cur_seqlen_neg_const).get_node_shared_ptr());
            neg_cur_seqlen = cur_neg_node->cast_vector<int64_t>()[0];
        }
        ov::Output<ov::Node> total_seqlen_output;
        if (pattern_map.count(total_seqlen) > 0) {
            total_seqlen_output = pattern_map.at(total_seqlen);
        }
        ov::Output<ov::Node> seqlens_k_output;
        if (pattern_map.count(seqlens_k) > 0) {
            seqlens_k_output = pattern_map.at(seqlens_k);
        }
        ov::Output<ov::Node> concat_kvlen_output;
        if (pattern_map.count(concat_kv_len) > 0) {
            concat_kvlen_output = pattern_map.at(concat_kv_len);
        }
        std::shared_ptr<CachedNodes> cache;
        if (total_seqlen_output.get_node()) {
            if (const auto it = m_cache.find(total_seqlen_output.get_node_shared_ptr()); it != m_cache.end()) {
                cache = it->second;
            }
        }
        if (!cache && seqlens_k_output.get_node()) {
            if (const auto it = m_cache.find(seqlens_k_output.get_node_shared_ptr()); it != m_cache.end()) {
                cache = it->second;
            }
        }
        if (!cache && concat_kvlen_output.get_node()) {
            if (const auto it = m_cache.find(concat_kvlen_output.get_node_shared_ptr()); it != m_cache.end()) {
                cache = it->second;
            }
        }
        if (!cache) {
            cache = std::make_shared<CachedNodes>();
        }
        if (total_seqlen_output.get_node() && !cache->total_seqlen) {
            cache->total_seqlen = total_seqlen_output.get_node_shared_ptr();
            if (!cache->present_kv_len.get_node()) {
                cache->present_kv_len = total_seqlen_output;
            }
            m_cache.try_emplace(total_seqlen_output.get_node_shared_ptr(), cache);
        }
        if (seqlens_k_output.get_node() && !cache->seqlens_k) {
            cache->seqlens_k = seqlens_k_output.get_node_shared_ptr();
            if (!cache->present_kv_len.get_node()) {
                cache->present_kv_len = pattern_map.at(real_seqlens);  //  skip reshape to pass correct shape-inder dependency
            }
            m_cache.try_emplace(seqlens_k_output.get_node_shared_ptr(), cache);
        }
        if (concat_kvlen_output.get_node()) {
            if (!cache->present_kv_len.get_node()) {
                cache->present_kv_len = concat_kvlen_output;
            }
            m_cache.try_emplace(concat_kvlen_output.get_node_shared_ptr(), cache);
        }

        if (is_slice_concat) {  // original dynamic pattern
            auto slice_node = ov::as_type_ptr<ov::op::v8::Slice>(pattern_map.at(slice).get_node_shared_ptr());
            auto concat_node = ov::as_type_ptr<ov::op::v0::Concat>(pattern_map.at(concat).get_node_shared_ptr());
            auto slice_stop_node = pattern_map.at(past_seqlen_actual).get_node_shared_ptr();
            auto slice_axis_node = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(slice_axis).get_node_shared_ptr());

            if (slice_axis_node->cast_vector<int64_t>()[0] != concat_node->get_axis()) {
                return false;
            }
            target_axis = concat_node->get_axis();

            if (cache->present_kv_len.get_node()) {
                seqlen_output = cache->present_kv_len;
            } else {
                seqlen_output = slice_stop_node;
                is_present_len = false;
            }
            node_infos = {slice_node, concat_node};
        } else {
            auto scatter_axis_node = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(scatter_axis).get_node_shared_ptr());
            target_axis = scatter_axis_node->cast_vector<int64_t>()[0];
            auto update_node = ov::as_type_ptr<ov::op::v3::ScatterUpdate>(pattern_map.at(scatter_update).get_node_shared_ptr());
            pos_idx_output = pattern_map.at(pos_idx);
            node_infos.push_back(update_node);

            const auto split_node = ov::as_type_ptr<ov::op::v1::VariadicSplit>(kv_to_sdpa_input->get_node()->shared_from_this());
            if (split_node) {  // static pattern with variadic split
                is_update_split = true;
                if (kv_to_sdpa_input->get_index() != 0) {
                    return false;
                }

                auto split_axis_node = ov::as_type_ptr<ov::op::v0::Constant>(split_node->get_input_node_shared_ptr(1));
                auto split_lengths_node = ov::as_type_ptr<ov::op::v0::Concat>(split_node->get_input_node_shared_ptr(2));
                if (!split_axis_node || !split_lengths_node || split_lengths_node->get_input_size() != 2) {
                    return false;
                }

                if (target_axis != split_axis_node->cast_vector<int64_t>()[0]) {
                    return false;
                }

                auto split_tail_node = ov::as_type_ptr<ov::op::v0::Constant>(split_lengths_node->get_input_node_shared_ptr(1));
                if (!split_tail_node) {
                    return false;
                }
                const auto split_tail_values = split_tail_node->cast_vector<int64_t>();
                if (split_tail_values.size() != 1 || split_tail_values[0] != -1) {
                    return false;
                }

                auto split_present_output = split_lengths_node->input_value(0);
                if (const auto split_present_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(split_present_output.get_node_shared_ptr())) {
                    split_present_output = split_present_reshape->input_value(0);
                }
                if (const auto plen_shape = split_present_output.get_partial_shape(); plen_shape.is_dynamic() || shape_size(plen_shape.get_shape()) != 1) {
                    return false;
                }
                seqlen_output = split_present_output;

                if (split_node->get_output_size() != 2 || !split_node->output(1).get_target_inputs().empty()) {
                    return false;
                }
                node_infos.push_back(split_node);
                const auto split_output_consumers = split_node->output(0).get_target_inputs();
                if (split_output_consumers.size() != 1) {
                    return false;
                }
                kv_sdpa_input.emplace(*split_output_consumers.begin());
            } else {  // original static pattern
                if (past_output.get_partial_shape().is_dynamic() || new_token_shape.is_dynamic()) {
                    return false;
                }

                if (pattern_map.count(shifted_pos_idx) == 0) {
                    return false;
                }

                ov::Output<ov::Node> past_seqlen_output = pattern_map.at(past_seqlen_from_param).get_node_shared_ptr();
                // make sure idx_* starts from 0 so that the other input is the past_seqlen_node
                if (pattern_map.count(const_range_cur) > 0) {
                    ov::op::v0::Constant* idx_const = ov::as_type<ov::op::v0::Constant>(pattern_map.at(const_range_cur).get_node());
                    const auto idx_data = idx_const->cast_vector<int64_t>();
                    if (idx_data.size() < 1) {
                        return false;
                    }
                    for (size_t i = 0; i < idx_data.size(); ++i) {
                        if (idx_data[i] != static_cast<int64_t>(i)) {
                            return false;
                        }
                    }
                } else {
                    // should already be garuanteed
                }

                if (cache->present_kv_len.get_node()) {
                    seqlen_output = cache->present_kv_len;
                } else {
                    seqlen_output = past_seqlen_output;
                    is_present_len = false;
                }
            }
        }

        if (shapeof_axis && *shapeof_axis != target_axis) {
            return false;
        }
        if (neg_cur_seqlen && (new_token_shape[target_axis].is_dynamic() || new_token_shape[target_axis].get_length() * -1 != *neg_cur_seqlen)) {
            return false;
        }

        if (!kv_sdpa_input) {
            kv_sdpa_input.emplace(*kv_to_sdpa_input);
        }

        auto kv_sdpa_node = kv_sdpa_input->get_node()->shared_from_this();
        std::shared_ptr<op::SDPA> sdpa_node = ov::as_type_ptr<op::SDPA>(kv_sdpa_node);
        if (sdpa_node && kv_sdpa_input->get_index() != 1 && kv_sdpa_input->get_index() != 2) {
            return false;
        }
        if (transformation_callback(kv_sdpa_node)) {
            return false;
        }

        auto get_trimmed_mask = [&](const ov::Output<ov::Node>& full_mask, const ov::Dimension& cur_seqlen) -> std::shared_ptr<ov::Node> {
            if (cache->present_kv_len.get_node() && cur_seqlen.is_static()) {
                for (const auto& [len, old_mask, new_mask] : cache->trimmed_masks) {
                    if (len != cur_seqlen.get_length())
                        continue;
                    if (old_mask == full_mask) {
                        return new_mask;
                    }
                }
            }
            return {};
        };
        // mask trimming for pure-scatter_update case
        if (!is_slice_concat && !is_update_split && sdpa_node && sdpa_node->inputs().size() - sdpa_node->get_compression_inputs_num() >= 4 &&
            m_trimmed_masks.count(sdpa_node->input_value(3)) == 0) {
            const auto full_mask = sdpa_node->input_value(3);
            const auto& cur_seqlen = new_token_shape[target_axis];
            auto trimmed_mask = get_trimmed_mask(full_mask, cur_seqlen);
            if (!trimmed_mask) {
                const auto present_len_type = seqlen_output.get_element_type();
                ov::Output<ov::Node> present_len = cache->present_kv_len;
                if (present_len.get_node()) {
                    if (present_len.get_partial_shape().rank().get_length() > 1) {
                        present_len = std::make_shared<v1::Reshape>(present_len, v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}), false);
                    }
                } else {
                    OPENVINO_ASSERT(!is_present_len);
                    std::shared_ptr<ov::Node> cur_seqlen_node;
                    if (cur_seqlen.is_static()) {
                        cur_seqlen_node = v0::Constant::create(present_len_type, ov::Shape{1}, {cur_seqlen.get_length()});
                    } else {
                        const auto zero_without_shape = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
                        auto new_token_shape = std::make_shared<v3::ShapeOf>(new_token_output, present_len_type);
                        cur_seqlen_node = std::make_shared<v8::Gather>(new_token_shape,
                                                                       v0::Constant::create(ov::element::i64, ov::Shape{1}, {target_axis}),
                                                                       zero_without_shape);
                    }
                    present_len = std::make_shared<v1::Add>(seqlen_output.get_node_shared_ptr(), cur_seqlen_node);
                }
                const auto split_lengths =
                    std::make_shared<v0::Concat>(ov::OutputVector{present_len, v0::Constant::create(present_len_type, ov::Shape{1}, {-1})}, 0);
                const auto mask_split = std::make_shared<v1::VariadicSplit>(full_mask, v0::Constant::create(present_len_type, ov::Shape{}, {1}), split_lengths);
                trimmed_mask = mask_split;
                if (cache->present_kv_len.get_node() && cur_seqlen.is_static()) {
                    cache->trimmed_masks.push_back({cur_seqlen.get_length(), full_mask, trimmed_mask});
                }
            }
            GPU_DEBUG_TRACE_DETAIL << "statelesskv: mask-trim [" << full_mask.get_node()->get_friendly_name() << "] -> [" << trimmed_mask->get_friendly_name()
                                   << "]" << std::endl;
            sdpa_node->set_argument(3, trimmed_mask->output(0));
            m_trimmed_masks.insert(trimmed_mask->output(0));
        }

        std::string posidname;
        if (pos_idx_output.get_node()) {
            posidname = pos_idx_output.get_node()->get_friendly_name();
        }
        GPU_DEBUG_TRACE_DETAIL << "statelesskv " << (is_slice_concat ? "SC" : (is_update_split ? "US" : "U")) << ": [" << past_output.get_any_name() << "]["
                               << result_node->get_friendly_name() << "] " << (sdpa_node ? "sdpa" : "next") << ":" << kv_sdpa_node->get_friendly_name()
                               << " len:" << seqlen_output.get_node()->get_friendly_name() << "(" << (is_present_len ? "present" : "past")
                               << ") pos:" << posidname
                               << " clen:" << (cache->present_kv_len.get_node() ? cache->present_kv_len.get_node()->get_friendly_name() : "") << std::endl;
        std::shared_ptr<op::StatelessKV> stateless_kv;
        if (!pos_idx_output.get_node()) {
            stateless_kv = std::make_shared<op::StatelessKV>(past_output, new_token_output, seqlen_output, target_axis, is_present_len);
        } else {
            stateless_kv = std::make_shared<op::StatelessKV>(past_output, new_token_output, seqlen_output, pos_idx_output, target_axis, is_present_len);
        }
        stateless_kv->set_friendly_name(past_output.get_any_name() + "_stateless");
        ov::copy_runtime_info(node_infos, stateless_kv);
        stateless_kv->output(0).set_names(result_node->output(0).get_names());

        kv_sdpa_input->replace_source_output(stateless_kv->output(1));
        result_node->input(0).replace_source_output(stateless_kv->output(0));
        for (auto& input : other_inputs) {
            // for full static case, need to keep other inputs to use original full shape
            input.replace_source_output(stateless_kv->output(is_slice_concat || is_update_split ? 1 : 0));
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, "StatelessKVFusionMatcher");
    this->register_matcher(m, callback);
}

bool StatelessKVFusion::run_on_model(const std::shared_ptr<ov::Model>& m) {
    return pass::GraphRewrite::run_on_model(m);
}

StatelessKVFusion::StatelessKVFusion() {
    add_matcher<ov::intel_gpu::StatelessKVFusionMatcher>();
}

}  // namespace ov::intel_gpu
