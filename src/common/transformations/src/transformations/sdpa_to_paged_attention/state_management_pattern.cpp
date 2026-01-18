// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/sdpa_to_paged_attention/state_management_pattern.hpp"

#include <tuple>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

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
constexpr const char* NUM_K_HEADS = "num_k_heads";
constexpr const char* K_HEAD_SIZE = "k_head_size";
constexpr const char* NUM_V_HEADS = "num_v_heads";
constexpr const char* V_HEAD_SIZE = "v_head_size";
using namespace ov::pass;
using ov::OutputVector;

static std::tuple<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> general_alibi_pattern() {
    // Optional pattern to capture alibi slopes (based on pattern from bloom)
    auto general_alibi = any_input();
    auto general_sdpa_mask = wrap_type<v1::Multiply>({any_input(), general_alibi});  // apply input position_ids
    general_sdpa_mask = wrap_type<v1::Reshape>({general_sdpa_mask, any_input()});
    general_sdpa_mask = wrap_type<v1::Reshape>({general_sdpa_mask, any_input()});
    general_sdpa_mask = wrap_type<v1::Select>({any_input(), any_input(), general_sdpa_mask});
    return {general_alibi, general_sdpa_mask};
}

static std::tuple<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> jais_13b_alibi_pattern() {
    auto jais_13b_alibi = any_input();
    auto mirroring_abs = wrap_type<v0::Abs>({any_input()});
    auto unsqueeze = wrap_type<v0::Unsqueeze>({mirroring_abs, any_input()});
    auto jais_alibi_mask = wrap_type<v1::Multiply>({jais_13b_alibi, unsqueeze});
    jais_alibi_mask = wrap_type<v3::Broadcast>({jais_alibi_mask, any_input()});
    jais_alibi_mask = wrap_type<v0::Unsqueeze>({jais_alibi_mask, any_input()});
    jais_alibi_mask = wrap_type<v1::Add>({any_input(), jais_alibi_mask});
    return {jais_13b_alibi, jais_alibi_mask};
}

static std::tuple<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> baichuan2_13b_alibi_pattern() {
    auto baichuan2_alibi = any_input();
    // this slice expected to be replaced with Slice(alibi_const, start {1, 1}, stop {2, 2}, step {1, 1}, axes{1, 2});
    auto alibi_slice_to_replace =
        wrap_type<v8::Slice>({baichuan2_alibi, any_input(), any_input(), any_input(), any_input()});
    auto alibi_path = wrap_type<v3::ShapeOf>({alibi_slice_to_replace});
    alibi_path = wrap_type<v8::Gather>({alibi_path, any_input(), any_input()});
    alibi_path = wrap_type<v0::Concat>({any_input(), any_input(), alibi_path});
    alibi_path = wrap_type<v3::Broadcast>({any_input(), alibi_path});
    alibi_path = wrap_type<v0::Convert>({alibi_path});
    alibi_path = wrap_type<v1::Multiply>({alibi_path, any_input()});
    alibi_path = wrap_type<v1::Subtract>({any_input(), alibi_path});
    alibi_path = wrap_type<v1::Select>({any_input(), any_input(), alibi_path});
    auto alibi_unsqueeze = wrap_type<v0::Unsqueeze>({alibi_slice_to_replace, any_input()});
    alibi_path = wrap_type<v1::Add>({alibi_path, alibi_unsqueeze});
    auto mul = wrap_type<v1::Multiply>({any_input(), any_input()});
    alibi_path = wrap_type<v8::Slice>({alibi_path, mul, any_input(), any_input(), any_input()});
    return {baichuan2_alibi, alibi_path};
}

static std::shared_ptr<ov::Node> handle_general_alibi(const std::shared_ptr<ov::Node>& matched_general_alibi_slopes) {
    std::shared_ptr<ov::Node> res_alibi_slopes =
        std::make_shared<v1::Reshape>(matched_general_alibi_slopes,
                                      v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}),
                                      false);
    if (res_alibi_slopes->get_element_type() != ov::element::f32) {
        res_alibi_slopes = std::make_shared<v0::Convert>(res_alibi_slopes, ov::element::f32);
    }

    return res_alibi_slopes;
}

static std::shared_ptr<ov::Node> handle_jais_13b_alibi(const std::shared_ptr<ov::Node>& matched_jais_13b_alibi_slopes) {
    // At the beginning, handling of jais13's alibi is the same as the general case
    std::shared_ptr<ov::Node> res_alibi_slopes = handle_general_alibi(matched_jais_13b_alibi_slopes);

    // For now there's no such case with Alibi slopes being not a Constant,
    // however that may change in the future. That is why the presence of
    // Abs is the main sign of the Jais-like topology, thus we need to multiply
    // by -1. If we encounter the Alibi being a constant, we may do the additional
    // checking of the values to be negative and, if it fails, we won't multiply
    // the values by -1.
    if (auto alibi_constant = ov::as_type_ptr<v0::Constant>(matched_jais_13b_alibi_slopes)) {
        auto alibi_constant_values = alibi_constant->cast_vector<float>();
        bool all_values_nagative =
            std::all_of(alibi_constant_values.begin(), alibi_constant_values.end(), [&](float value) {
                return value < 0.0;
            });

        if (all_values_nagative) {
            res_alibi_slopes =
                std::make_shared<v1::Multiply>(res_alibi_slopes,
                                               v0::Constant::create(res_alibi_slopes->get_element_type(), {}, {-1}));
        }
    } else {
        res_alibi_slopes =
            std::make_shared<v1::Multiply>(res_alibi_slopes,
                                           v0::Constant::create(res_alibi_slopes->get_element_type(), {}, {-1}));
    }

    return res_alibi_slopes;
}

static std::shared_ptr<ov::Node> handle_baichuan2_13b_alibi(
    /* >>> alibi = np.reshape(alibi, (40, 4096, 4096))
       >>> print(alibi[0][:][:])
       [['0' '-inf' '-inf' ... '-inf' '-inf' '-inf']
        ['0' '0.839844' '-inf' ... '-inf' '-inf' '-inf']
        ['0' '0.839844' '1.67969' ... '-inf' '-inf' '-inf']
        ...
        ['0' '0.839844' '1.67969' ... '3440' '-inf' '-inf']
        ['0' '0.839844' '1.67969' ... '3440' '3440' '-inf']
        ['0' '0.839844' '1.67969' ... '3440' '3440' '3440']]
       >>> print(alibi[1][:][:])
       [['0' '-inf' '-inf' ... '-inf' '-inf' '-inf']
        ['0' '0.707031' '-inf' ... '-inf' '-inf' '-inf']
        ['0' '0.707031' '1.41406' ... '-inf' '-inf' '-inf']
        ...
        ['0' '0.707031' '1.41406' ... '2896' '-inf' '-inf']
        ['0' '0.707031' '1.41406' ... '2896' '2896' '-inf']
        ['0' '0.707031' '1.41406' ... '2896' '2896' '2896']]

        etc.

        Slicing from {1, 1} to {2, 2} gives us the expected alibi slope constant to pass it to PagedAttention:
        >>> print(alibi[0][1][1])
        0.839844
        >>> print(line1[1][1][1])
        0.707031

        ALibi slopes constant's shape is [40, 4096, 4096]
        Slicing means that we take only 1 value from each 4096 x 4096 matrix here
        The resulting constant will be [40, 1, 1]
        After that we need to insert Reshape to get the expected rank = 1 (shape [40])
    */
    const std::shared_ptr<ov::Node>& matched_baichuan2_13b_alibi_slopes) {
    std::shared_ptr<ov::Node> res_alibi_slopes = matched_baichuan2_13b_alibi_slopes;

    auto start = v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 1});
    auto stop = v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 2});
    auto step = v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 1});
    auto axes = v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 2});
    // the Slice to extract the correct values
    res_alibi_slopes = std::make_shared<v8::Slice>(res_alibi_slopes, start, stop, step, axes);
    res_alibi_slopes = std::make_shared<v1::Reshape>(res_alibi_slopes,
                                                     v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}),
                                                     false);
    if (res_alibi_slopes->get_element_type() != ov::element::f32) {
        res_alibi_slopes = std::make_shared<v0::Convert>(res_alibi_slopes, ov::element::f32);
    }

    return res_alibi_slopes;
}

static std::tuple<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> phi3_sliding_window_pattern() {
    auto offset = wrap_type<v0::Constant>();
    auto t196 = wrap_type<v1::Add>({any_input(), offset});
    auto t197 = ov::pass::pattern::optional<v0::Convert>(t196);
    auto t200 = wrap_type<ov::op::v4::Range>({t197, any_input(), any_input()});
    auto t201 = wrap_type<v0::Unsqueeze>({t200, any_input()});
    auto t202 = wrap_type<v1::GreaterEqual>({any_input(), t201});
    auto t208 = wrap_type<v1::Select>({t202, any_input(), any_input()});
    auto t209 = wrap_type<v1::Subtract>({any_input(), t208});
    auto t210 = ov::pass::pattern::optional<v0::Convert>(t209);
    auto t211 = wrap_type<v1::Select>({t210, any_input(), any_input()});
    auto t213 = wrap_type<v0::Unsqueeze>({t211, any_input()});
    auto t214 = wrap_type<v0::Unsqueeze>({t213, any_input()});
    auto t218 = wrap_type<v3::Broadcast>({t214, any_input()});
    auto t219 = wrap_type<v1::Select>({any_input(), any_input(), t218});
    auto mask = wrap_type<v8::Slice>({t219, any_input(), any_input(), any_input(), any_input()});
    return {mask, offset};
}

static std::tuple<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> gpt_oss_sliding_window_pattern() {
    auto q_idx = any_input();
    auto kv_idx = any_input();

    auto kv_idx_opt_conv = ov::pass::pattern::optional<v0::Convert>(kv_idx);

    auto offset = wrap_type<v0::Constant>();

    auto add = wrap_type<v1::Add>({q_idx, offset});
    auto greater = wrap_type<v1::Greater>({kv_idx_opt_conv, add});
    auto bitwise_and = wrap_type<v13::BitwiseAnd>({any_input(), greater});
    auto bitwise_and_1 = wrap_type<v13::BitwiseAnd>({bitwise_and, any_input()});
    auto bitwise_and_2 = wrap_type<v13::BitwiseAnd>({any_input(), bitwise_and_1});
    auto bitwise_and_3 = wrap_type<v13::BitwiseAnd>({bitwise_and_2, any_input()});
    auto broadcast = wrap_type<v3::Broadcast>({bitwise_and_3, any_input()});
    auto select = wrap_type<v1::Select>({broadcast, any_input(), any_input()});
    auto mask = wrap_type<v8::Slice>({select, any_input(), any_input(), any_input(), any_input()});

    return {mask, offset};
}

// Exactly copied the function from another file. Maybe should be moved to some general file
static std::shared_ptr<v0::Parameter> setName(std::shared_ptr<v0::Parameter> node, const std::string& name) {
    // Set name for both node and output tensor (should be only one tensor, and any other names will be overriden by a
    // given single name)
    node->set_friendly_name(name);
    OPENVINO_ASSERT(node->get_output_size() == 1);
    node->get_output_tensor(0).set_names({name});
    return node;
}

typedef std::
    tuple<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>>
        node_tuple;

static node_tuple kv_read_and_concat(ov::Output<ov::Node> kv_current) {
    auto kv_past_var = wrap_type<ov::op::v6::ReadValue>({any_input()});
    auto kv_past_par = wrap_type<v0::Parameter>();
    auto kv_past =
        std::make_shared<Or>(OutputVector{wrap_type<v8::Gather>({kv_past_var, any_input(), any_input()}), kv_past_par});
    kv_past = std::make_shared<Or>(
        OutputVector{kv_past,
                     wrap_type<v1::Transpose>({kv_past, any_input()})});  // Transpose is used when kv-cache is stored
                                                                          // in a not usual layout, example: bloom
    auto kv_current2 = any_input();
    auto kv_current_reshaped = wrap_type<v1::Reshape>({kv_current2, any_input()});
    auto kv_concat =
        wrap_type<v0::Concat>({kv_past, std::make_shared<Or>(OutputVector{kv_current_reshaped, kv_current})});
    return node_tuple(kv_past_par, kv_current2, kv_current_reshaped, kv_concat);
}

static ov::Dimension extract_num_kv_heads(const std::shared_ptr<ov::Node>& unsqueeze_pattern,
                                          const ov::Dimension& default_heads_num,
                                          const ov::pass::pattern::PatternValueMap& pattern_map) {
    // Deduce number of k/v heads from Unsqueeze-Broadcast-Reshape (UBR pattern, if present)
    // pattern that appears in case of MQA/GQA.
    // In case if UBR pattern doesn't appear, the default number of heads is used passed as default_heads_num.
    if (pattern_map.find(unsqueeze_pattern) != pattern_map.end()) {
        // based on unsqueeze index determine the dimension that will be broadcased
        // if there is no expected dimension for any reason, return dynamic dimension
        auto unsqueeze = pattern_map.at(unsqueeze_pattern).get_node_shared_ptr();
        auto shape = unsqueeze->get_output_partial_shape(0);
        auto rank = shape.rank();
        if (rank.is_dynamic()) {
            return ov::Dimension();
        }
        auto axis = unsqueeze->get_input_node_ptr(1);
        auto constant = ov::as_type<v0::Constant>(axis);
        if (!constant) {
            return ov::Dimension();
        }
        if (ov::shape_size(constant->get_output_shape(0)) != 1) {  // it should be only one axis
            return ov::Dimension();
        }
        auto first_element = constant->cast_vector<int64_t>(1)[0];
        if (first_element == 0 ||
            first_element == -rank.get_length()) {  // there should be at least one dimension to the left
            return ov::Dimension();
        }
        // In some cases of MQA, where KV cache is stored as 3D tensor there is no dimension that corresponds to
        // num kv heads in KV tensor (because it is 1 and can be not exposed). Hence we should look at the
        // first_element - 1 axis first, if it is static then it is our number of heads, if it is not staic,
        // then the number of heads is 1, and Broadcast implements pure MQA logic within a single dimension.
        return shape[first_element - 1].is_static() ? shape[first_element - 1] : ov::Dimension(1);
    } else {
        return default_heads_num;
    }
};

ov::pass::StateManagementPattern::StateManagementPattern(
    ParameterVector& kv_parameters,
    ParameterVector& model_wide_params,
    ParameterVector& parameters_to_remove,
    int& layer_index,
    Output<Node> max_context_len,
    ParameterVector& block_indices_inputs_for_each_layer,
    ResultVector& score_results,
    bool use_per_layer_block_indices_inputs,
    bool use_score_outputs,
    bool allow_cache_rotation,
    bool allow_score_aggregation,
    bool allow_xattention,
    bool allow_adaptive_rkv,
    ParameterVector& rotated_block_indices_inputs_for_each_layer,
    ParameterVector& rotation_deltas_inputs_for_each_layer,
    ParameterVector& xattention_threshold_inputs_for_each_layer,
    ParameterVector& adaptive_rkv_diversity_block_set_indices_inputs_for_each_layer,
    ParameterVector& adaptive_rkv_diversity_block_set_indices_begins_inputs_for_each_layer,
    ResultVector& adaptive_rkv_diversity_results,
    const std::map<std::string, std::shared_ptr<op::v0::Parameter>>& optional_model_wide_params) {
    MATCHER_SCOPE(StateManagementPattern);

    auto k_current = any_input();
    std::shared_ptr<ov::Node> k_past_par, k_current2, k_concat, k_current_reshaped;
    std::tie(k_past_par, k_current2, k_current_reshaped, k_concat) = kv_read_and_concat(k_current);

    auto v_current = any_input();
    std::shared_ptr<ov::Node> v_past_par, v_current2, v_concat, v_current_reshaped;
    std::tie(v_past_par, v_current2, v_current_reshaped, v_concat) = kv_read_and_concat(v_current);

    // There are models where K and V merged into a single tensor and splited apart after K/V past and current
    // concatenation The following part in the pattern covers this case.
    // TODO: Consider not specifying VariadicSplit as an input for Concat, it is not really used in the pattern, but
    // just sets more strict requirement for the graph. The risk with not specifying VariadicSplit is that it can be
    // ambiguous which part the matcher should take: KV merged part or where K and V are separate, requires experiments.
    auto qkv_current_split_node = wrap_type<v1::VariadicSplit>({any_input(), any_input(), any_input()});
    qkv_current_split_node->set_output_size(2);
    auto kv_current = qkv_current_split_node->output(1);
    std::shared_ptr<ov::Node> kv_past_par, kv_current2, kv_concat, kv_current_reshaped;
    std::tie(kv_past_par, kv_current2, kv_current_reshaped, kv_concat) = kv_read_and_concat(kv_current);
    auto kv_concat_split = wrap_type<v1::VariadicSplit>({kv_concat, any_input(), any_input()});
    kv_concat_split->set_output_size(2);

    k_concat = std::make_shared<Or>(OutputVector{kv_concat_split->output(0), k_concat});
    v_concat = std::make_shared<Or>(OutputVector{kv_concat_split->output(1), v_concat});

    auto kv_shaping = [=](const std::shared_ptr<Node>& kv_concat, std::shared_ptr<Node>& unsqueeze) {
        // Return unsqeeze (return param) to deduce number of kv heads in
        // the place where they are being broadcases in case of GQA and MQ
        auto interim = wrap_type<v1::StridedSlice>({kv_concat, any_input(), any_input(), any_input()});
        interim = wrap_type<v1::StridedSlice>({interim, any_input(), any_input(), any_input()});
        unsqueeze = wrap_type<v0::Unsqueeze>({std::make_shared<Or>(OutputVector{kv_concat, interim}), any_input()});
        interim = wrap_type<v1::StridedSlice>({unsqueeze, any_input(), any_input(), any_input()});
        interim = wrap_type<v1::StridedSlice>({interim, any_input(), any_input(), any_input()});
        interim = wrap_type<v3::Broadcast>({std::make_shared<Or>(OutputVector{unsqueeze, interim}), any_input()});
        interim = std::make_shared<Or>(OutputVector{wrap_type<v1::Reshape>({interim, any_input()}),
                                                    interim});  // Reshape is missing sometimes in MQA case
        return interim;
    };

    std::shared_ptr<Node> k_heads_unsqueeze;
    std::shared_ptr<Node> v_heads_unsqueeze;
    auto k_shaped = kv_shaping(k_concat, k_heads_unsqueeze);
    auto v_shaped = kv_shaping(v_concat, v_heads_unsqueeze);

    auto k_simply_shaped = wrap_type<v1::Reshape>({k_concat, any_input()});
    auto v_simply_shaped = wrap_type<v1::Reshape>({v_concat, any_input()});

    auto k_order = any_input();
    auto v_order = any_input();

    // KV-path may already have Transposes that will be rewritten based on PA KV inputs required layout
    auto k_shaped_transposed =
        wrap_type<v1::Transpose>({std::make_shared<Or>(OutputVector{k_concat, k_shaped}), k_order});
    auto v_shaped_transposed =
        wrap_type<v1::Transpose>({std::make_shared<Or>(OutputVector{v_concat, v_shaped}), v_order});

    // Optional pattern to capture alibi slopes (based on pattern from bloom)
    std::shared_ptr<ov::Node> general_alibi, general_alibi_mask;
    std::tie(general_alibi, general_alibi_mask) = general_alibi_pattern();

    // For Jais (Jais-13b has a different pattern and handling of alibi slopes)
    std::shared_ptr<ov::Node> jais_13b_alibi, jais_alibi_mask;
    std::tie(jais_13b_alibi, jais_alibi_mask) = jais_13b_alibi_pattern();

    // Baichuan2 13b case
    std::shared_ptr<ov::Node> baichuan2_13b_alibi, baichuan2_13b_alibi_mask;
    std::tie(baichuan2_13b_alibi, baichuan2_13b_alibi_mask) = baichuan2_13b_alibi_pattern();

    // Phi3-xxx-instruct case
    std::shared_ptr<ov::Node> phi3_mask, phi3_offset;
    std::tie(phi3_mask, phi3_offset) = phi3_sliding_window_pattern();

    // gpt-oss case
    std::shared_ptr<ov::Node> gpt_oss_mask, gpt_oss_offset;
    std::tie(gpt_oss_mask, gpt_oss_offset) = gpt_oss_sliding_window_pattern();

    // Scale's shape limitations according to SDPA specification
    auto scale_predicate = [=](const Output<Node>& output) -> bool {
        return output.get_partial_shape() == ov::PartialShape{} ||
               (output.get_partial_shape() == ov::PartialShape{1} && output.get_partial_shape()[0] == 1);
    };

    auto q = any_input();
    auto scale_input = any_input(scale_predicate);
    auto sinks = any_input(ov::pass::pattern::has_static_shape() && ov::pass::pattern::rank_equals(4));

    auto k_to_sdpa = std::make_shared<Or>(OutputVector{k_concat, k_shaped, k_shaped_transposed, k_simply_shaped});
    auto v_to_sdpa = std::make_shared<Or>(OutputVector{v_concat, v_shaped, v_shaped_transposed, v_simply_shaped});

    auto mask_to_sdpa = std::make_shared<Or>(OutputVector{phi3_mask,
                                                          general_alibi_mask,
                                                          jais_alibi_mask,
                                                          baichuan2_13b_alibi_mask,
                                                          gpt_oss_mask,
                                                          any_input()});

    auto sdpa_with_4_inputs = wrap_type<v13::ScaledDotProductAttention>({q, k_to_sdpa, v_to_sdpa, mask_to_sdpa});
    auto sdpa_with_5_inputs =
        wrap_type<v13::ScaledDotProductAttention>({q, k_to_sdpa, v_to_sdpa, mask_to_sdpa, scale_input});
    auto sdpa_with_6_inputs =
        wrap_type<v13::ScaledDotProductAttention>({q, k_to_sdpa, v_to_sdpa, mask_to_sdpa, scale_input, sinks});

    auto sdpa_variants = std::make_shared<Or>(OutputVector{sdpa_with_4_inputs, sdpa_with_5_inputs, sdpa_with_6_inputs});

    ov::matcher_pass_callback callback = [=,
                                          &kv_parameters,
                                          &model_wide_params,
                                          &parameters_to_remove,
                                          &block_indices_inputs_for_each_layer,
                                          &score_results,
                                          &layer_index,
                                          &rotated_block_indices_inputs_for_each_layer,
                                          &rotation_deltas_inputs_for_each_layer,
                                          &xattention_threshold_inputs_for_each_layer,
                                          &adaptive_rkv_diversity_block_set_indices_inputs_for_each_layer,
                                          &adaptive_rkv_diversity_block_set_indices_begins_inputs_for_each_layer,
                                          &adaptive_rkv_diversity_results](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& real_q = pattern_map.at(q);

        auto sdpa_node = pattern_map
                             .at(pattern_map.count(sdpa_with_4_inputs)   ? sdpa_with_4_inputs
                                 : pattern_map.count(sdpa_with_5_inputs) ? sdpa_with_5_inputs
                                                                         : sdpa_with_6_inputs)
                             .get_node();

        auto k_head_size_dim = sdpa_node->get_input_tensor(1).get_partial_shape()[-1];  // E from SDPA spec.
        auto v_head_size_dim = sdpa_node->get_input_tensor(2)
                                   .get_partial_shape()[-1];  // Ev from SDPA spec. (in common case may not match E)
        OPENVINO_ASSERT((k_head_size_dim.is_static() && v_head_size_dim.is_static()),
                        "k/v_head_size dimensions have to be static.");
        auto k_head_size = k_head_size_dim.get_length();
        auto v_head_size = v_head_size_dim.get_length();

        auto num_k_heads_dim = extract_num_kv_heads(k_heads_unsqueeze,
                                                    sdpa_node->get_input_tensor(1).get_partial_shape()[-3],
                                                    pattern_map);
        auto num_v_heads_dim = extract_num_kv_heads(v_heads_unsqueeze,
                                                    sdpa_node->get_input_tensor(2).get_partial_shape()[-3],
                                                    pattern_map);
        OPENVINO_ASSERT((num_k_heads_dim.is_static() && num_v_heads_dim.is_static()),
                        "num_k/v_head dimensions have to be static.");
        auto num_k_heads = num_k_heads_dim.get_length();
        auto num_v_heads = num_v_heads_dim.get_length();

        std::string layer_index_str = std::to_string(layer_index);
        auto k_parameter = setName(std::make_shared<v0::Parameter>(element::dynamic, ov::PartialShape::dynamic(4)),
                                   "key_cache." + layer_index_str);
        auto v_parameter = setName(std::make_shared<v0::Parameter>(element::dynamic, ov::PartialShape::dynamic(4)),
                                   "value_cache." + layer_index_str);

        layer_index += 1;
        kv_parameters.push_back(k_parameter);
        kv_parameters.push_back(v_parameter);
        auto kv_transpose_order = v0::Constant::create(element::i64, Shape{4}, {0, 2, 1, 3});

        auto q_transpose = std::make_shared<v1::Transpose>(real_q, kv_transpose_order);
        auto q_reshape =
            std::make_shared<v1::Reshape>(q_transpose, v0::Constant::create(element::i64, Shape{2}, {0, -1}), true);

        ov::Output<ov::Node> k_target_layout, v_target_layout;
        if (pattern_map.count(qkv_current_split_node)) {
            // Fast track for merged K/V caches, based on the currently observed models topologies we don't need to
            // change layout and there is no point in the graph where it is in 4D. So `else` branch below is not
            // applicable for this case. + std::to_string(layer_index - 1)
            auto qkv_split = pattern_map.at(qkv_current_split_node).get_node_shared_ptr();
            // TODO: Consider handling Q part as well as KV here, requires more changes in the code and sets
            // VariadicSplit before Concat as essential part of the pattern
            auto kv_split_part = qkv_split->output(1);
            auto real_kv_concat_split = pattern_map.at(kv_concat_split).get_node_shared_ptr();
            // Reaply VariadicSplit from the model after the Concat with KV merged tensor to current KV merged tensor
            // before the Concat
            auto kv_current_split = real_kv_concat_split->clone_with_new_inputs(
                {kv_split_part, real_kv_concat_split->input_value(1), real_kv_concat_split->input_value(2)});
            // Under assumption that K and V parts go in order: K part first, and then V part. Theoretically they can be
            // swapped.
            // TODO: Need more code to track the swapped variant.
            k_target_layout = kv_current_split->output(0);
            v_target_layout = kv_current_split->output(1);
        } else {
            // takes option that has 4D instead of fine-grained Reshape analysis
            // it avoids complication in the pattern, but we don't really have many options
            auto take_4d = [=](const std::shared_ptr<Node>& option1,
                               const std::shared_ptr<Node>& option2,
                               const std::shared_ptr<Node>& option3) {
                if (pattern_map.find(option1) != pattern_map.end() &&
                    pattern_map.at(option1).get_partial_shape().rank().get_length() == 4) {
                    return pattern_map.at(option1);
                } else if (pattern_map.at(option2).get_partial_shape().rank().get_length() == 4) {
                    return pattern_map.at(option2);
                } else {
                    return pattern_map.at(option3);
                }
            };

            auto real_k = take_4d(k_current, k_current_reshaped, k_current2);
            auto real_v = take_4d(v_current, v_current_reshaped, v_current2);

            std::shared_ptr<Node> k_transpose_order = kv_transpose_order;
            if (pattern_map.find(k_order) !=
                pattern_map
                    .end()) {  // reapply transpose found in the graph by manipulating of indices of our Transpose
                k_transpose_order = std::make_shared<v8::Gather>(pattern_map.at(k_order),
                                                                 kv_transpose_order,
                                                                 v0::Constant::create(element::i64, Shape{}, {0}));
            }
            k_target_layout = std::make_shared<v1::Transpose>(real_k, k_transpose_order);
            std::shared_ptr<Node> v_transpose_order = kv_transpose_order;
            if (pattern_map.find(v_order) !=
                pattern_map
                    .end()) {  // reapply transpose found in the graph by manipulating of indices of our Transpose
                v_transpose_order = std::make_shared<v8::Gather>(pattern_map.at(v_order),
                                                                 kv_transpose_order,
                                                                 v0::Constant::create(element::i64, Shape{}, {0}));
            }
            v_target_layout = std::make_shared<v1::Transpose>(real_v, v_transpose_order);
        }

        auto k_reshape =
            std::make_shared<v1::Reshape>(k_target_layout, v0::Constant::create(element::i64, Shape{2}, {0, -1}), true);
        auto v_reshape =
            std::make_shared<v1::Reshape>(v_target_layout, v0::Constant::create(element::i64, Shape{2}, {0, -1}), true);

        std::shared_ptr<ov::Node> scale;
        if (pattern_map.count(scale_input)) {
            scale = pattern_map.at(scale_input).get_node_shared_ptr();
            if (pattern_map.at(scale_input).get_partial_shape().rank() != 0) {
                scale = std::make_shared<v15::Squeeze>(scale);
            }
        } else {
            auto real_q_ps = real_q.get_partial_shape();

            bool rank_is_static = real_q_ps.rank().is_static();
            if (rank_is_static && real_q_ps[real_q_ps.rank().get_length() - 1].is_static()) {
                auto hidden_dim_len = static_cast<float>(real_q_ps[real_q_ps.rank().get_length() - 1].get_length());
                scale = v0::Constant::create(element::f32, Shape{}, {1.0 / std::sqrt(hidden_dim_len)});
            } else {
                // most likely `scale` below will always be a constant in real inference, but dynamic dimension
                // propagation may not always derive it as a constant. That's why a sub-graph computing `scale` is built
                // instead of just a constant node representing one of the dimensions.
                auto hidden_shape = std::make_shared<v3::ShapeOf>(real_q);
                auto hidden_dim = std::make_shared<v8::Gather>(hidden_shape,
                                                               v0::Constant::create(element::i64, Shape{}, {-1}),
                                                               v0::Constant::create(element::i64, Shape{}, {0}));
                scale = std::make_shared<v1::Divide>(
                    v0::Constant::create(element::f32, Shape{}, {1}),
                    std::make_shared<v0::Sqrt>(std::make_shared<v0::Convert>(hidden_dim, element::f32)));
            }
        }

        std::shared_ptr<Node> alibi_slopes;
        if (pattern_map.find(general_alibi) != pattern_map.end()) {
            alibi_slopes = handle_general_alibi(pattern_map.at(general_alibi).get_node_shared_ptr());
        } else if (pattern_map.find(jais_13b_alibi) != pattern_map.end()) {
            alibi_slopes = handle_jais_13b_alibi(pattern_map.at(jais_13b_alibi).get_node_shared_ptr());
        } else if (pattern_map.find(baichuan2_13b_alibi) != pattern_map.end()) {
            alibi_slopes = handle_baichuan2_13b_alibi(pattern_map.at(baichuan2_13b_alibi).get_node_shared_ptr());
        } else {
            alibi_slopes = v0::Constant::create(element::f32, Shape{0}, {});
        }

        OutputVector pa_arguments = {q_reshape, k_reshape, v_reshape, k_parameter, v_parameter};
        pa_arguments.insert(pa_arguments.end(), model_wide_params.begin(), model_wide_params.end());

        std::shared_ptr<Node> sliding_window;
        if (pattern_map.count(phi3_offset)) {
            auto offset = pattern_map.at(phi3_offset).get_node_shared_ptr();
            if (offset->get_element_type() != element::i32) {
                offset = std::make_shared<v0::Convert>(offset, element::i32);
            }
            sliding_window = std::make_shared<v1::Subtract>(v0::Constant::create(element::i32, Shape{}, {2}), offset);
        } else if (pattern_map.count(gpt_oss_offset)) {
            auto offset = pattern_map.at(gpt_oss_offset).get_node_shared_ptr();
            if (pattern_map.at(gpt_oss_offset).get_partial_shape().rank() != 0) {
                offset = std::make_shared<v15::Squeeze>(offset);
            }
            if (offset->get_element_type() != element::i32) {
                offset = std::make_shared<v0::Convert>(offset, element::i32);
            }
            sliding_window = std::make_shared<v1::Multiply>(offset, v0::Constant::create(element::i32, Shape{}, {-1}));
        } else {
            sliding_window = v0::Constant::create(element::i32, Shape{}, {0});
        }

        std::initializer_list<std::shared_ptr<Node>> additional_params = {scale,
                                                                          sliding_window,
                                                                          alibi_slopes,
                                                                          max_context_len.get_node_shared_ptr()};
        pa_arguments.insert(pa_arguments.end(), additional_params.begin(), additional_params.end());

        if (use_per_layer_block_indices_inputs) {
            auto block_indices = setName(std::make_shared<v0::Parameter>(element::i32, PartialShape{-1}),
                                         "block_indices." + std::to_string(layer_index - 1));
            pa_arguments.insert(pa_arguments.begin() + 7, block_indices);
            block_indices_inputs_for_each_layer.push_back(block_indices);
        }

        if (allow_score_aggregation) {
            OPENVINO_ASSERT(
                optional_model_wide_params.find("score_aggregation_window") != optional_model_wide_params.end(),
                "No score_aggregation_window input found. For using score aggregation mode, the model have to contain "
                "an additional input (Parameter) called score_aggregation_window.");
            pa_arguments.insert(pa_arguments.end(), optional_model_wide_params.at("score_aggregation_window"));
        } else {
            pa_arguments.insert(pa_arguments.end(), v0::Constant::create(element::i32, Shape{0}, {}));
        }
        OPENVINO_ASSERT(pa_arguments.size() == 14);

        if (allow_cache_rotation) {
            OPENVINO_ASSERT(
                optional_model_wide_params.find("model_rotation_trig_lut") != optional_model_wide_params.end(),
                "No model_rotation_trig_lut input found. For using cache rotation, the model have to contain "
                "an additional input (Parameter) called model_rotation_trig_lut.");
            auto rotated_block_indices = setName(std::make_shared<v0::Parameter>(element::i32, PartialShape{-1}),
                                                 "rotated_block_indices." + std::to_string(layer_index - 1));
            auto rotation_deltas = setName(std::make_shared<v0::Parameter>(element::i32, PartialShape{-1, -1}),
                                           "rotation_deltas." + std::to_string(layer_index - 1));

            pa_arguments.insert(pa_arguments.begin() + 14, rotated_block_indices);
            pa_arguments.insert(pa_arguments.begin() + 15, rotation_deltas);
            pa_arguments.insert(pa_arguments.begin() + 16, optional_model_wide_params.at("model_rotation_trig_lut"));

            rotated_block_indices_inputs_for_each_layer.push_back(rotated_block_indices);
            rotation_deltas_inputs_for_each_layer.push_back(rotation_deltas);
        } else {
            auto rotated_block_indices = v0::Constant::create(element::i32, Shape{0}, {});
            auto rotation_deltas = v0::Constant::create(element::i32, Shape{0}, {});
            pa_arguments.insert(pa_arguments.begin() + 14, rotated_block_indices);
            pa_arguments.insert(pa_arguments.begin() + 15, rotation_deltas);
            pa_arguments.insert(pa_arguments.begin() + 16, v0::Constant::create(element::f32, Shape{0}, {}));
        }

        OPENVINO_ASSERT(pa_arguments.size() == 17);
        if (allow_xattention) {
            OPENVINO_ASSERT(
                optional_model_wide_params.find("xattention_block_size") != optional_model_wide_params.end(),
                "No xattention_block_size input found. For using XAttention, the model have to contain "
                "an additional input (Parameter) called xattention_block_size.");
            OPENVINO_ASSERT(optional_model_wide_params.find("xattention_stride") != optional_model_wide_params.end(),
                            "No xattention_stride input found. For using XAttention, the model have to contain "
                            "an additional input (Parameter) called xattention_stride.");
            auto xattention_threshold = setName(std::make_shared<v0::Parameter>(element::f32, PartialShape{-1}),
                                                "xattention_threshold." + std::to_string(layer_index - 1));
            pa_arguments.insert(pa_arguments.begin() + 17, xattention_threshold);
            pa_arguments.insert(pa_arguments.begin() + 18, optional_model_wide_params.at("xattention_block_size"));
            pa_arguments.insert(pa_arguments.begin() + 19, optional_model_wide_params.at("xattention_stride"));
            xattention_threshold_inputs_for_each_layer.push_back(xattention_threshold);
        } else {
            auto xattention_threshold = v0::Constant::create(element::f32, Shape{0}, {});
            pa_arguments.insert(pa_arguments.begin() + 17, xattention_threshold);
            pa_arguments.insert(pa_arguments.begin() + 18, v0::Constant::create(element::i32, Shape{}, {0}));
            pa_arguments.insert(pa_arguments.begin() + 19, v0::Constant::create(element::i32, Shape{}, {0}));
        }

        // For now we haven't seen sinks in any other model than gpt-oss, so taking -3 is generally safe
        // as there's going to be num_q_heads at -3.
        if (pattern_map.count(sinks)) {
            const auto& sinks_val = pattern_map.at(sinks);
            if (sinks_val.get_partial_shape()[-3] == real_q.get_partial_shape()[-3]) {
                pa_arguments.insert(pa_arguments.begin() + 20, sinks_val.get_node_shared_ptr());
            } else {
                pa_arguments.insert(pa_arguments.begin() + 20,
                                    v0::Constant::create(real_q.get_element_type(), Shape{0, 0, 0, 0}, {}));
            }
        } else {
            pa_arguments.insert(pa_arguments.begin() + 20,
                                v0::Constant::create(real_q.get_element_type(), Shape{0, 0, 0, 0}, {}));
        }

        OPENVINO_ASSERT(pa_arguments.size() == 21);

        if (allow_adaptive_rkv) {
            OPENVINO_ASSERT(
                optional_model_wide_params.find("adaptive_rkv_start_size") != optional_model_wide_params.end(),
                "No adaptive_rkv_start_size input found. For using Adaptive R-KV, the model have to contain "
                "an additional input (Parameter) called adaptive_rkv_start_size.");
            OPENVINO_ASSERT(
                optional_model_wide_params.find("adaptive_rkv_evictable_sizes") != optional_model_wide_params.end(),
                "No adaptive_rkv_evictable_sizes input found. For using Adaptive R-KV, the model have to contain "
                "an additional input (Parameter) called adaptive_rkv_evictable_sizes.");
            pa_arguments.insert(pa_arguments.begin() + 21, optional_model_wide_params.at("adaptive_rkv_start_size"));
            pa_arguments.insert(pa_arguments.begin() + 22,
                                optional_model_wide_params.at("adaptive_rkv_evictable_sizes"));

            auto adaptive_rkv_diversity_block_set_indices =
                setName(std::make_shared<v0::Parameter>(element::i32, PartialShape{-1}),
                        "adaptive_rkv_diversity_block_set_indices." + std::to_string(layer_index - 1));
            pa_arguments.insert(pa_arguments.begin() + 23, adaptive_rkv_diversity_block_set_indices);
            adaptive_rkv_diversity_block_set_indices_inputs_for_each_layer.push_back(
                adaptive_rkv_diversity_block_set_indices);

            auto adaptive_rkv_diversity_block_set_indices_begins =
                setName(std::make_shared<v0::Parameter>(element::i32, PartialShape{-1}),
                        "adaptive_rkv_diversity_block_set_indices_begins." + std::to_string(layer_index - 1));
            pa_arguments.insert(pa_arguments.begin() + 24, adaptive_rkv_diversity_block_set_indices_begins);
            adaptive_rkv_diversity_block_set_indices_begins_inputs_for_each_layer.push_back(
                adaptive_rkv_diversity_block_set_indices_begins);

        } else {
            pa_arguments.insert(pa_arguments.begin() + 21, v0::Constant::create(element::i32, Shape{}, {0}));
            pa_arguments.insert(pa_arguments.begin() + 22, v0::Constant::create(element::i32, Shape{0}, {}));
            pa_arguments.insert(pa_arguments.begin() + 23, v0::Constant::create(element::i32, Shape{0}, {}));
            pa_arguments.insert(pa_arguments.begin() + 24, v0::Constant::create(element::i32, Shape{0}, {}));
        }
        OPENVINO_ASSERT(pa_arguments.size() == 25);

        auto paged_attention = std::make_shared<ov::op::PagedAttentionExtension>(pa_arguments);
        paged_attention->get_rt_info()[NUM_K_HEADS] = num_k_heads;
        paged_attention->get_rt_info()[K_HEAD_SIZE] = k_head_size;
        paged_attention->get_rt_info()[NUM_V_HEADS] = num_v_heads;
        paged_attention->get_rt_info()[V_HEAD_SIZE] = v_head_size;

        // The output shape of PagedAttention will be converted to [batch, 1, head_num, head_size_v], the head_size_v
        // may be different from head_size_q/head_size_k. The head_size_v could be got from the shape of value input
        auto hidden_dim_v = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(v_target_layout),
                                                         v0::Constant::create(element::i64, Shape{}, {-1}),
                                                         v0::Constant::create(element::i64, Shape{}, {0}));

        auto pa_shape = std::make_shared<v0::Concat>(
            OutputVector{
                v0::Constant::create(element::i64, Shape{1}, {0}),
                v0::Constant::create(element::i64, Shape{1}, {1}),
                v0::Constant::create(element::i64, Shape{1}, {-1}),
                std::make_shared<v0::Unsqueeze>(hidden_dim_v, v0::Constant::create(element::i64, Shape{}, {0})),
            },
            0);
        auto pa_reshape = std::make_shared<v1::Reshape>(paged_attention->output(0), pa_shape, true);
        auto pa_transpose = std::make_shared<v1::Transpose>(pa_reshape, kv_transpose_order);
        if (use_score_outputs) {
            auto score_result = std::make_shared<v0::Result>(paged_attention->output(1));
            score_result->get_output_tensor(0).set_names({"scores." + std::to_string(layer_index - 1)});
            score_results.push_back(score_result);
        }

        if (allow_adaptive_rkv) {
            auto similarity_result = std::make_shared<v0::Result>(paged_attention->output(2));
            similarity_result->get_output_tensor(0).set_names(
                {"adaptive_rkv_diversity." + std::to_string(layer_index - 1)});
            adaptive_rkv_diversity_results.push_back(similarity_result);
        }

        // TODO: Complete this part to work with stateless models as well as will stateful
        //  def add_kv_parameter(past_node):
        //      if past_node.get_type_info().name == 'Parameter':
        //          parameters_to_remove.append(past_node)

        //  add_kv_parameter(mapping[k_gather])
        //  add_kv_parameter(mapping[v_gather])

        if (pattern_map.find(v_past_par) != pattern_map.end()) {
            auto param = ov::as_type_ptr<v0::Parameter>(pattern_map.at(v_past_par).get_node_shared_ptr());
            if (param) {
                return false;
            }
            parameters_to_remove.push_back(param);
        }

        if (pattern_map.find(k_past_par) != pattern_map.end()) {
            auto param = ov::as_type_ptr<v0::Parameter>(pattern_map.at(k_past_par).get_node_shared_ptr());
            if (param) {
                return false;
            }
            parameters_to_remove.push_back(param);
        }

        pa_transpose->set_friendly_name(sdpa_node->get_friendly_name());
        replace_node(m.get_match_root(), pa_transpose);
        return true;
    };

    auto m = std::make_shared<Matcher>(sdpa_variants, matcher_name);
    register_matcher(m, callback);
}
