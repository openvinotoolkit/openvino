// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/gemma4_mtp_state_management_pattern.hpp"

#include <cmath>
#include <tuple>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"

namespace pattern = ov::pass::pattern;
using ov::pass::pattern::any_input;
using ov::pass::pattern::Matcher;
using ov::pass::pattern::wrap_type;
using ov::pass::pattern::op::Or;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v8 = ov::op::v8;
namespace v13 = ov::op::v13;
namespace v15 = ov::op::v15;

using ov::OutputVector;
using ov::pass::paged_attention::PaParams;

namespace {

constexpr const char* NUM_K_HEADS = "num_k_heads";
constexpr const char* K_HEAD_SIZE = "k_head_size";
constexpr const char* NUM_V_HEADS = "num_v_heads";
constexpr const char* V_HEAD_SIZE = "v_head_size";

std::tuple<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> sliding_window_pattern() {
    auto offset = wrap_type<v0::Constant>();
    auto abs = wrap_type<v0::Abs>({any_input()});
    auto band = wrap_type<v1::LessEqual>({abs, offset});
    auto bitwise_and = wrap_type<v13::BitwiseAnd>({any_input(), band});
    auto bitwise_and_1 = wrap_type<v13::BitwiseAnd>({bitwise_and, any_input()});
    auto bitwise_and_2 = wrap_type<v13::BitwiseAnd>({any_input(), bitwise_and_1});
    auto bitwise_and_3 = wrap_type<v13::BitwiseAnd>({bitwise_and_2, any_input()});
    auto broadcast = wrap_type<v3::Broadcast>({bitwise_and_3, any_input()});
    auto mask = pattern::optional<v8::Slice>({broadcast, any_input(), any_input(), any_input(), any_input()});

    return {mask, offset};
}

// K/V arrive as a [B, Hkv, L, S] model input
std::shared_ptr<ov::Node> kv_input_path(const std::shared_ptr<ov::Node>& kv_param) {
    auto unsqueeze = wrap_type<v0::Unsqueeze>({kv_param, any_input()});
    auto broadcast = wrap_type<v3::Broadcast>({unsqueeze, any_input()});
    return wrap_type<v1::Reshape>({broadcast, any_input()});
}

std::shared_ptr<ov::Node> kv_placeholder(const ov::Output<ov::Node>& total_token_count,
                                         ov::element::Type q_type,
                                         int64_t width) {
    // scalar [] -> [1] = {B_token}
    auto count =
        std::make_shared<v0::Unsqueeze>(total_token_count, v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    // [1] = {width}
    auto width_dim = v0::Constant::create(ov::element::i64, ov::Shape{1}, {width});
    // [2] = {B_token, width}
    auto shape = std::make_shared<v0::Concat>(OutputVector{count, width_dim}, 0);
    // scalar 0 -> [B_token, width] of zeros
    return std::make_shared<v3::Broadcast>(v0::Constant::create(q_type, ov::Shape{}, {0}), shape);
}

}  // namespace

ov::pass::Gemma4MTPStateManagementPattern::Gemma4MTPStateManagementPattern(
    PaParams& pa_params,
    std::unordered_set<std::string>& params_to_remove) {
    MATCHER_SCOPE(Gemma4MTPStateManagementPattern);

    auto borrowed_kv_input = pattern::rank_equals(4) && pattern::has_static_dims({1, 3});
    auto k_param = wrap_type<v0::Parameter>(borrowed_kv_input);
    auto v_param = wrap_type<v0::Parameter>(borrowed_kv_input);
    auto k_to_sdpa = kv_input_path(k_param);
    auto v_to_sdpa = kv_input_path(v_param);
    auto scale_input = any_input(pattern::shape_matches("[]") || pattern::shape_matches("[1]"));

    std::shared_ptr<ov::Node> mask_to_sdpa, sliding_window_offset;
    std::tie(mask_to_sdpa, sliding_window_offset) = sliding_window_pattern();
    mask_to_sdpa = mask_to_sdpa | any_input();

    auto sdpa_variants =
        wrap_type<v13::ScaledDotProductAttention>({any_input(), k_to_sdpa, v_to_sdpa, mask_to_sdpa}) |
        wrap_type<v13::ScaledDotProductAttention>({any_input(), k_to_sdpa, v_to_sdpa, mask_to_sdpa, scale_input});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS, &pa_params, &params_to_remove](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto sdpa_node = ov::as_type_ptr<v13::ScaledDotProductAttention>(m.get_match_root());
        if (!sdpa_node) {
            return false;
        }

        auto k_src = ov::as_type_ptr<v0::Parameter>(pattern_map.at(k_param).get_node_shared_ptr());
        auto v_src = ov::as_type_ptr<v0::Parameter>(pattern_map.at(v_param).get_node_shared_ptr());
        if (!k_src || !v_src) {
            return false;
        }

        const auto& k_ps = k_src->get_partial_shape();
        const auto& v_ps = v_src->get_partial_shape();
        const auto num_k_heads = k_ps[1].get_length();
        const auto k_head_size = k_ps[3].get_length();
        const auto num_v_heads = v_ps[1].get_length();
        const auto v_head_size = v_ps[3].get_length();

        // Layers reading the same K/V input share one cache.
        const auto k_input = k_src->get_output_tensor(0).get_any_name();
        const auto v_input = v_src->get_output_tensor(0).get_any_name();
        const auto& k_cache_name =
            m_input_to_cache.try_emplace(k_input, "key_cache." + std::to_string(m_layer_index)).first->second;
        const auto& v_cache_name =
            m_input_to_cache.try_emplace(v_input, "value_cache." + std::to_string(m_layer_index)).first->second;
        auto k_cache_param = pa_params.add(k_cache_name, ov::element::dynamic, ov::PartialShape::dynamic(4));
        auto v_cache_param = pa_params.add(v_cache_name, ov::element::dynamic, ov::PartialShape::dynamic(4));
        enable_keep_const_precision(k_cache_param);
        enable_keep_const_precision(v_cache_param);
        params_to_remove.insert(k_input);
        params_to_remove.insert(v_input);

        auto bhls_to_blhs = v0::Constant::create(element::i64, Shape{4}, {0, 2, 1, 3});
        auto real_q = sdpa_node->input_value(0);
        auto q_transpose = std::make_shared<v1::Transpose>(real_q, bhls_to_blhs);
        auto q_to_pa =
            std::make_shared<v1::Reshape>(q_transpose, v0::Constant::create(element::i64, Shape{2}, {0, -1}), true);

        // With no K/V projections the current token is taken from the cache itself; PagedAttention then
        // sees it as the one new token, so it has to be excluded from past_lens.
        auto past_lens_minus_one = std::make_shared<v1::Subtract>(
            pa_params["past_lens"],
            v0::Constant::create(pa_params["past_lens"]->get_output_element_type(0), Shape{}, {1}));

        auto total_token_count = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(q_to_pa, ov::element::i64),
                                                              v0::Constant::create(ov::element::i64, ov::Shape{}, {0}),
                                                              v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));

        auto k_to_pa = kv_placeholder(total_token_count, q_to_pa->get_element_type(), num_k_heads * k_head_size);
        auto v_to_pa = num_v_heads * v_head_size == num_k_heads * k_head_size
                           ? k_to_pa
                           : kv_placeholder(total_token_count, q_to_pa->get_element_type(), num_v_heads * v_head_size);

        std::shared_ptr<Node> scale;
        if (pattern_map.count(scale_input)) {
            scale = pattern_map.at(scale_input).get_node_shared_ptr();
            if (pattern_map.at(scale_input).get_partial_shape().rank() != 0) {
                scale = std::make_shared<v15::Squeeze>(scale);
            }
        } else {
            scale = v0::Constant::create(element::f32, Shape{}, {1.0 / std::sqrt(static_cast<float>(k_head_size))});
        }

        auto sliding_window = [&]() -> std::shared_ptr<Node> {
            if (!pattern_map.count(sliding_window_offset)) {
                return v0::Constant::create(element::i32, Shape{}, {0});
            }
            const auto& matched_offset = pattern_map.at(sliding_window_offset);
            std::shared_ptr<Node> offset = matched_offset.get_node_shared_ptr();
            if (matched_offset.get_partial_shape().rank() != 0) {
                offset = std::make_shared<v15::Squeeze>(offset);
            }
            if (offset->get_element_type() != element::i32) {
                offset = std::make_shared<v0::Convert>(offset, element::i32);
            }
            return offset;
        }();

        OutputVector pa_arguments = {q_to_pa, k_to_pa, v_to_pa, k_cache_param, v_cache_param};
        pa_arguments.push_back(past_lens_minus_one);
        pa_arguments.push_back(pa_params["subsequence_begins"]);
        pa_arguments.push_back(pa_params["block_indices"]);
        pa_arguments.push_back(pa_params["block_indices_begins"]);
        pa_arguments.push_back(scale);
        pa_arguments.push_back(sliding_window);
        pa_arguments.push_back(v0::Constant::create(element::f32, Shape{0}, {}));  // alibi_slopes
        pa_arguments.push_back(pa_params["max_context_len"]);
        pa_arguments.push_back(v0::Constant::create(element::i32, Shape{0}, {}));  // score_aggregation_window
        pa_arguments.push_back(v0::Constant::create(element::i32, Shape{0}, {}));  // rotated_block_indices
        pa_arguments.push_back(v0::Constant::create(element::i32, Shape{0}, {}));  // rotation_deltas
        pa_arguments.push_back(v0::Constant::create(element::f32, Shape{0}, {}));  // rotation_trig_lut
        pa_arguments.push_back(v0::Constant::create(element::f32, Shape{0}, {}));  // xattention_threshold
        pa_arguments.push_back(v0::Constant::create(element::i32, Shape{}, {0}));  // xattention_block_size
        pa_arguments.push_back(v0::Constant::create(element::i32, Shape{}, {0}));  // xattention_stride
        pa_arguments.push_back(v0::Constant::create(real_q.get_element_type(), Shape{0, 0, 0, 0}, {}));  // sinks
        pa_arguments.push_back(v0::Constant::create(element::i32, Shape{}, {0}));  // adaptive_rkv_start_size
        pa_arguments.push_back(v0::Constant::create(element::i32, Shape{0}, {}));  // adaptive_rkv_evictable_sizes
        pa_arguments.push_back(v0::Constant::create(element::i32, Shape{0}, {}));  // adaptive_rkv_div_set_indices
        pa_arguments.push_back(v0::Constant::create(element::i32, Shape{0}, {}));  // adaptive_rkv_div_set_begins
        pa_arguments.push_back(v0::Constant::create(element::i32, Shape{0}, {}));  // token_type_ids
        pa_arguments.push_back(v0::Constant::create(element::u8, Shape{0}, {}));   // qq_bias
        pa_arguments.push_back(v0::Constant::create(element::i32, Shape{0}, {}));  // qq_bias_begins
        OPENVINO_ASSERT(pa_arguments.size() == 28);

        auto paged_attention =
            std::make_shared<ov::op::PagedAttentionExtension>(pa_arguments, /*write_kv_cache=*/false);
        paged_attention->get_rt_info()[NUM_K_HEADS] = num_k_heads;
        paged_attention->get_rt_info()[K_HEAD_SIZE] = k_head_size;
        paged_attention->get_rt_info()[NUM_V_HEADS] = num_v_heads;
        paged_attention->get_rt_info()[V_HEAD_SIZE] = v_head_size;

        // [B_token, H * Sv] back to the SDPA layout [B_token, H, 1, Sv].
        auto pa_reshape = std::make_shared<v1::Reshape>(
            paged_attention->output(0),
            v0::Constant::create(element::i64, Shape{4}, std::vector<int64_t>{0, 1, -1, v_head_size}),
            true);
        auto pa_transpose = std::make_shared<v1::Transpose>(pa_reshape, bhls_to_blhs);

        pa_transpose->set_friendly_name(sdpa_node->get_friendly_name());
        replace_node(m.get_match_root(), pa_transpose);
        m_layer_index += 1;
        return true;
    };

    auto m = std::make_shared<Matcher>(sdpa_variants, matcher_name);
    register_matcher(m, callback);
}
