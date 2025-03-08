// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/sdpa_to_paged_attention/state_management_pattern.hpp"

#include <tuple>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op;
using namespace ov::pass;
using ov::OutputVector;

static std::tuple<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> general_alibi_pattern() {
    // Optional pattern to capture alibi slopes (based on pattern from bloom)
    auto general_alibi = pattern::any_input();
    auto general_sdpa_mask =
        pattern::wrap_type<v1::Multiply>({pattern::any_input(), general_alibi});  // apply input position_ids
    general_sdpa_mask = pattern::wrap_type<v1::Reshape>({general_sdpa_mask, pattern::any_input()});
    general_sdpa_mask = pattern::wrap_type<v1::Reshape>({general_sdpa_mask, pattern::any_input()});
    general_sdpa_mask = pattern::wrap_type<v1::Select>({pattern::any_input(), pattern::any_input(), general_sdpa_mask});
    return {general_alibi, general_sdpa_mask};
}

static std::tuple<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> jais_13b_alibi_pattern() {
    auto jais_13b_alibi = pattern::any_input();
    auto mirroring_abs = pattern::wrap_type<v0::Abs>({pattern::any_input()});
    auto unsqueeze = pattern::wrap_type<v0::Unsqueeze>({mirroring_abs, pattern::any_input()});
    auto jais_alibi_mask = pattern::wrap_type<v1::Multiply>({jais_13b_alibi, unsqueeze});
    jais_alibi_mask = pattern::wrap_type<v3::Broadcast>({jais_alibi_mask, pattern::any_input()});
    jais_alibi_mask = pattern::wrap_type<v0::Unsqueeze>({jais_alibi_mask, pattern::any_input()});
    jais_alibi_mask = pattern::wrap_type<v1::Add>({pattern::any_input(), jais_alibi_mask});
    return {jais_13b_alibi, jais_alibi_mask};
}

static std::tuple<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> baichuan2_13b_alibi_pattern() {
    auto baichuan2_alibi = pattern::any_input();
    // this slice expected to be replaced with Slice(alibi_const, start {1, 1}, stop {2, 2}, step {1, 1}, axes{1, 2});
    auto alibi_slice_to_replace = pattern::wrap_type<v8::Slice>(
        {baichuan2_alibi, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto alibi_path = pattern::wrap_type<v3::ShapeOf>({alibi_slice_to_replace});
    alibi_path = pattern::wrap_type<v8::Gather>({alibi_path, pattern::any_input(), pattern::any_input()});
    alibi_path = pattern::wrap_type<v0::Concat>({pattern::any_input(), pattern::any_input(), alibi_path});
    alibi_path = pattern::wrap_type<v3::Broadcast>({pattern::any_input(), alibi_path});
    alibi_path = pattern::wrap_type<v0::Convert>({alibi_path});
    alibi_path = pattern::wrap_type<v1::Multiply>({alibi_path, pattern::any_input()});
    alibi_path = pattern::wrap_type<v1::Subtract>({pattern::any_input(), alibi_path});
    alibi_path = pattern::wrap_type<v1::Select>({pattern::any_input(), pattern::any_input(), alibi_path});
    auto alibi_unsqueeze = pattern::wrap_type<v0::Unsqueeze>({alibi_slice_to_replace, pattern::any_input()});
    alibi_path = pattern::wrap_type<v1::Add>({alibi_path, alibi_unsqueeze});
    auto mul = pattern::wrap_type<v1::Multiply>({pattern::any_input(), pattern::any_input()});
    alibi_path = pattern::wrap_type<v8::Slice>(
        {alibi_path, mul, pattern::any_input(), pattern::any_input(), pattern::any_input()});
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
    auto kv_past_var = pattern::wrap_type<v6::ReadValue>({pattern::any_input()});
    auto kv_past_par = pattern::wrap_type<v0::Parameter>();
    auto kv_past = std::make_shared<pattern::op::Or>(
        OutputVector{pattern::wrap_type<v8::Gather>({kv_past_var, pattern::any_input(), pattern::any_input()}),
                     kv_past_par});
    kv_past = std::make_shared<pattern::op::Or>(
        OutputVector{kv_past,
                     pattern::wrap_type<v1::Transpose>(
                         {kv_past, pattern::any_input()})});  // Transpose is used when kv-cache is stored in a not
                                                              // usual layout, example: bloom
    auto kv_current2 = pattern::any_input();
    auto kv_current_reshaped = pattern::wrap_type<v1::Reshape>({kv_current2, pattern::any_input()});
    auto kv_concat = pattern::wrap_type<v0::Concat>(
        {kv_past, std::make_shared<pattern::op::Or>(OutputVector{kv_current_reshaped, kv_current})});
    return node_tuple(kv_past_par, kv_current2, kv_current_reshaped, kv_concat);
}

ov::pass::StateManagementPattern::StateManagementPattern(ParameterVector& kv_parameters,
                                                         ParameterVector& model_remaining_params,
                                                         const std::shared_ptr<ov::op::v0::Constant>& sliding_window,
                                                         ParameterVector& parameters_to_remove,
                                                         int& layer_index,
                                                         Output<Node> max_context_len,
                                                         ParameterVector& block_indices_inputs_for_each_layer,
                                                         ResultVector& score_results,
                                                         bool use_per_layer_block_indices_inputs,
                                                         bool use_score_outputs,
                                                         bool allow_cache_rotation,
                                                         ParameterVector& rotated_block_indices_inputs_for_each_layer,
                                                         ParameterVector& rotation_deltas_inputs_for_each_layer,
                                                         std::shared_ptr<op::v0::Parameter> model_rotation_trig_lut) {
    MATCHER_SCOPE(StateManagementPattern);

    auto k_current = pattern::any_input();
    std::shared_ptr<ov::Node> k_past_par, k_current2, k_concat, k_current_reshaped;
    std::tie(k_past_par, k_current2, k_current_reshaped, k_concat) = kv_read_and_concat(k_current);

    auto v_current = pattern::any_input();
    std::shared_ptr<ov::Node> v_past_par, v_current2, v_concat, v_current_reshaped;
    std::tie(v_past_par, v_current2, v_current_reshaped, v_concat) = kv_read_and_concat(v_current);

    // There are models where K and V merged into a single tensor and splited apart after K/V past and current
    // concatenation The following part in the pattern covers this case.
    // TODO: Consider not specifying VariadicSplit as an input for Concat, it is not really used in the pattern, but
    // just sets more strict requirement for the graph. The risk with not specifying VariadicSplit is that it can be
    // ambiguous which part the matcher should take: KV merged part or where K and V are separate, requires experiments.
    auto qkv_current_split_node =
        pattern::wrap_type<v1::VariadicSplit>({pattern::any_input(), pattern::any_input(), pattern::any_input()});
    qkv_current_split_node->set_output_size(2);
    auto kv_current = qkv_current_split_node->output(1);
    std::shared_ptr<ov::Node> kv_past_par, kv_current2, kv_concat, kv_current_reshaped;
    std::tie(kv_past_par, kv_current2, kv_current_reshaped, kv_concat) = kv_read_and_concat(kv_current);
    auto kv_concat_split =
        pattern::wrap_type<v1::VariadicSplit>({kv_concat, pattern::any_input(), pattern::any_input()});
    kv_concat_split->set_output_size(2);

    k_concat = std::make_shared<pattern::op::Or>(OutputVector{kv_concat_split->output(0), k_concat});
    v_concat = std::make_shared<pattern::op::Or>(OutputVector{kv_concat_split->output(1), v_concat});

    auto kv_shaping = [=](const std::shared_ptr<Node>& kv_concat, std::shared_ptr<Node>& unsqueeze) {
        // Return unsqeeze (return param) to deduce number of kv heads in
        // the place where they are being broadcases in case of GQA and MQ
        auto interim = pattern::wrap_type<v1::StridedSlice>(
            {kv_concat, pattern::any_input(), pattern::any_input(), pattern::any_input()});
        interim = pattern::wrap_type<v1::StridedSlice>(
            {interim, pattern::any_input(), pattern::any_input(), pattern::any_input()});
        unsqueeze = pattern::wrap_type<v0::Unsqueeze>(
            {std::make_shared<pattern::op::Or>(OutputVector{kv_concat, interim}), pattern::any_input()});
        interim = pattern::wrap_type<v1::StridedSlice>(
            {unsqueeze, pattern::any_input(), pattern::any_input(), pattern::any_input()});
        interim = pattern::wrap_type<v1::StridedSlice>(
            {interim, pattern::any_input(), pattern::any_input(), pattern::any_input()});
        interim = pattern::wrap_type<v3::Broadcast>(
            {std::make_shared<pattern::op::Or>(OutputVector{unsqueeze, interim}), pattern::any_input()});
        interim = std::make_shared<pattern::op::Or>(
            OutputVector{pattern::wrap_type<v1::Reshape>({interim, pattern::any_input()}),
                         interim});  // Reshape is missing sometimes in MQA case
        return interim;
    };

    std::shared_ptr<Node> k_heads_unsqueeze;
    std::shared_ptr<Node> v_heads_unsqueeze;
    auto k_shaped = kv_shaping(k_concat, k_heads_unsqueeze);
    auto v_shaped = kv_shaping(v_concat, v_heads_unsqueeze);

    auto k_simply_shaped = pattern::wrap_type<v1::Reshape>({k_concat, pattern::any_input()});
    auto v_simply_shaped = pattern::wrap_type<v1::Reshape>({v_concat, pattern::any_input()});

    auto k_order = pattern::any_input();
    auto v_order = pattern::any_input();

    // KV-path may already have Transposes that will be rewritten based on PA KV inputs required layout
    auto k_shaped_transposed = pattern::wrap_type<v1::Transpose>(
        {std::make_shared<pattern::op::Or>(OutputVector{k_concat, k_shaped}), k_order});
    auto v_shaped_transposed = pattern::wrap_type<v1::Transpose>(
        {std::make_shared<pattern::op::Or>(OutputVector{v_concat, v_shaped}), v_order});

    // Optional pattern to capture alibi slopes (based on pattern from bloom)
    std::shared_ptr<ov::Node> general_alibi, general_alibi_mask;
    std::tie(general_alibi, general_alibi_mask) = general_alibi_pattern();

    // For Jais (Jais-13b has a different pattern and handling of alibi slopes)
    std::shared_ptr<ov::Node> jais_13b_alibi, jais_alibi_mask;
    std::tie(jais_13b_alibi, jais_alibi_mask) = jais_13b_alibi_pattern();

    // Baichuan2 13b case
    std::shared_ptr<ov::Node> baichuan2_13b_alibi, baichuan2_13b_alibi_mask;
    std::tie(baichuan2_13b_alibi, baichuan2_13b_alibi_mask) = baichuan2_13b_alibi_pattern();

    auto q = pattern::any_input();
    auto scale_input = pattern::any_input();

    auto k_to_sdpa =
        std::make_shared<pattern::op::Or>(OutputVector{k_concat, k_shaped, k_shaped_transposed, k_simply_shaped});
    auto v_to_sdpa =
        std::make_shared<pattern::op::Or>(OutputVector{v_concat, v_shaped, v_shaped_transposed, v_simply_shaped});
    auto mask_to_sdpa = std::make_shared<pattern::op::Or>(
        OutputVector{general_alibi_mask, jais_alibi_mask, baichuan2_13b_alibi_mask, pattern::any_input()});

    auto sdpa_with_4_inputs =
        pattern::wrap_type<v13::ScaledDotProductAttention>({q, k_to_sdpa, v_to_sdpa, mask_to_sdpa});
    auto sdpa_with_5_inputs =
        pattern::wrap_type<v13::ScaledDotProductAttention>({q, k_to_sdpa, v_to_sdpa, mask_to_sdpa, scale_input});

    auto sdpa_variants = std::make_shared<pattern::op::Or>(OutputVector{sdpa_with_4_inputs, sdpa_with_5_inputs});

    ov::matcher_pass_callback callback = [=,
                                          &kv_parameters,
                                          &model_remaining_params,
                                          &sliding_window,
                                          &parameters_to_remove,
                                          &block_indices_inputs_for_each_layer,
                                          &score_results,
                                          &layer_index,
                                          &rotated_block_indices_inputs_for_each_layer,
                                          &rotation_deltas_inputs_for_each_layer](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto real_q = pattern_map.at(q);

        auto sdpa_node =
            pattern_map.at(pattern_map.count(sdpa_with_4_inputs) ? sdpa_with_4_inputs : sdpa_with_5_inputs).get_node();
        // E and Ev are from the SDPA specification at
        // https://docs.openvino.ai/2025/documentation/openvino-ir-format/operation-sets/operation-specs/sequence/scaled-dot-product-attention.html
        auto E = sdpa_node->get_input_tensor(1).get_partial_shape()[-1];
        auto Ev = sdpa_node->get_input_tensor(2).get_partial_shape()[-1];  // in common case may not match E

        auto extract_num_kv_heads = [=, &pattern_map](std::shared_ptr<Node> unsqueeze,
                                                      const Dimension& default_heads_num) {
            // Deduce number of k/v heads from Unsqueeze-Broadcast-Reshape (UBR pattern, if present)
            // pattern that appears in case of MQA/GQA.
            // In case if UBR pattern doesn't appear, the default number of heads is used passed as default_heads_num.
            if (pattern_map.find(unsqueeze) != pattern_map.end()) {
                // based on unsqueeze index determine the dimension that will be broadcased
                // if there is no expected dimension for any reason, return dynamic dimension
                unsqueeze = pattern_map.at(unsqueeze).get_node_shared_ptr();
                auto shape = unsqueeze->get_output_partial_shape(0);
                auto rank = shape.rank();
                if (rank.is_dynamic()) {
                    return ov::Dimension();
                }
                rank = rank.get_length();
                auto axis = unsqueeze->input_value(1).get_node_shared_ptr();
                auto constant = ov::as_type_ptr<ov::op::v0::Constant>(axis);
                if (!constant) {
                    return ov::Dimension();
                }
                auto data = constant->cast_vector<int64_t>();
                if (data.size() != 1) {  // it should be only one axis
                    return ov::Dimension();
                }
                auto first_element = data[0];
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

        auto num_k_heads =
            extract_num_kv_heads(k_heads_unsqueeze, sdpa_node->get_input_tensor(1).get_partial_shape()[-3]);
        auto num_v_heads =
            extract_num_kv_heads(v_heads_unsqueeze, sdpa_node->get_input_tensor(2).get_partial_shape()[-3]);
        const ov::element::Type kv_cache_type = real_q.get_element_type();
        std::string layer_index_str = std::to_string(layer_index);
        auto k_parameter = setName(std::make_shared<v0::Parameter>(kv_cache_type, PartialShape{-1, num_k_heads, E}),
                                   std::string("key_cache.") + std::to_string(layer_index));
        auto v_parameter = setName(std::make_shared<v0::Parameter>(kv_cache_type, PartialShape{-1, num_v_heads, Ev}),
                                   std::string("value_cache.") + std::to_string(layer_index));
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
        pa_arguments.insert(pa_arguments.end(), model_remaining_params.begin(), model_remaining_params.end());
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

        OPENVINO_ASSERT(pa_arguments.size() == 13);

        if (allow_cache_rotation) {
            auto rotated_block_indices = setName(std::make_shared<v0::Parameter>(element::i32, PartialShape{-1}),
                                                 "rotated_block_indices." + std::to_string(layer_index - 1));
            auto rotation_deltas = setName(std::make_shared<v0::Parameter>(element::i32, PartialShape{-1, -1}),
                                           "rotation_deltas." + std::to_string(layer_index - 1));

            pa_arguments.insert(pa_arguments.begin() + 13, rotated_block_indices);
            pa_arguments.insert(pa_arguments.begin() + 14, rotation_deltas);
            pa_arguments.insert(pa_arguments.begin() + 15, model_rotation_trig_lut);

            rotated_block_indices_inputs_for_each_layer.push_back(rotated_block_indices);
            rotation_deltas_inputs_for_each_layer.push_back(rotation_deltas);
        }

        auto paged_attention = std::make_shared<ov::op::PagedAttentionExtension>(pa_arguments);

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

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sdpa_variants, matcher_name);
    register_matcher(m, callback);
}
