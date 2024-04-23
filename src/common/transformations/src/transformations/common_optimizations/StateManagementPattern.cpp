// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/StateManagementPattern.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/cc/pass/itt.hpp"
#include "openvino/op/paged_attention.hpp"

using namespace ov::op;

//TODO: think how to do it
// ov::OutputVector arguments_as_outputs(std::shared_ptr<ov::Node> arguments) {
//     ov::OutputVector outputs;
//     for (auto& argument : arguments) {
//         if () {
//             outputs.push_back(argument);
//         } else {
//             outputs.push_back(argument.)
//         }
//     }

//     return outputs;
// }

ov::pass::StateManagementPattern::StateManagementPattern(ParameterVector& kv_parameters,
                                                         const ParameterVector& model_remaining_params,
                                                         const std::shared_ptr<ov::op::v0::Constant>& sliding_window,
                                                         ParameterVector& parameters_to_remove,
                                                         std::vector<std::shared_ptr<Node>>& assignes_to_remove) {
    MATCHER_SCOPE(StateManagementPattern);

    auto k_past_var = pattern::wrap_type<v3::ReadValue>({pattern::any_input()});
    auto k_past_par = pattern::wrap_type<v3::ReadValue>();
    auto k_past = std::make_shared<pattern::op::Or>(OutputVector{pattern::wrap_type<v1::Gather>({k_past_par, pattern::any_input(), pattern::any_input()}), k_past_par});
    k_past = std::make_shared<pattern::op::Or>(OutputVector{k_past, pattern::wrap_type<v1::Transpose>({k_past, pattern::any_input()})}); //Transpose is used when kv-cache is stored in a not usual layout, example: bloom
    auto k_current = pattern::any_input();
    auto k_current2 = pattern::any_input();
    auto k_current_reshaped = pattern::wrap_type<v1::Reshape>({k_current2, pattern::any_input()});
    auto k_concat = pattern::wrap_type<v0::Concat>({k_past, std::make_shared<pattern::op::Or>(OutputVector{k_current_reshaped, k_current})});

    auto kv_shaping = [OV_CAPTURE_CPY_AND_THIS](std::shared_ptr<Node> kv_concat) {
        auto interim = pattern::wrap_type<v1::StridedSlice>({kv_concat, pattern::any_input(), pattern::any_input(), pattern::any_input()});
        interim = pattern::wrap_type<v1::StridedSlice>({interim, pattern::any_input(), pattern::any_input(), pattern::any_input()});
        auto unsqueeze = pattern::wrap_type<v0::Unsqueeze>({std::make_shared<pattern::op::Or>(OutputVector{kv_concat, interim}), pattern::any_input()});
        interim = pattern::wrap_type<v1::StridedSlice>({unsqueeze, pattern::any_input(), pattern::any_input(), pattern::any_input()});
        interim = pattern::wrap_type<v1::StridedSlice>({interim, pattern::any_input(), pattern::any_input(), pattern::any_input()});
        interim = pattern::wrap_type<v1::Broadcast>({std::make_shared<pattern::op::Or>(OutputVector{unsqueeze, interim}), pattern::any_input()});
        interim = pattern::wrap_type<v1::Reshape>({interim, pattern::any_input()});
        return interim;
    };

    auto v_past_var = pattern::wrap_type<v3::ReadValue>({pattern::any_input()});
    auto v_past_par = pattern::wrap_type<v0::Parameter>();
    auto v_past = std::make_shared<pattern::op::Or>(OutputVector{pattern::wrap_type<v1::Gather>({v_past_par, pattern::any_input(), pattern::any_input()}), v_past_par});
    v_past = std::make_shared<pattern::op::Or>(OutputVector{v_past, pattern::wrap_type<v1::Transpose>({v_past, pattern::any_input()})});
    auto v_current = pattern::any_input();
    auto v_current2 = pattern::any_input();
    auto v_current_reshaped = pattern::wrap_type<v1::Reshape>({v_current2, pattern::any_input()});
    auto v_concat = pattern::wrap_type<v0::Concat>({v_past, std::make_shared<pattern::op::Or>(OutputVector{v_current_reshaped, v_current})});

    auto k_shaped = kv_shaping(k_concat);
    auto v_shaped = kv_shaping(v_concat);

    auto k_simply_shaped = pattern::wrap_type<v1::Reshape>({k_concat, pattern::any_input()});
    auto v_simply_shaped = pattern::wrap_type<v1::Reshape>({v_concat, pattern::any_input()});

    auto k_order = pattern::any_input();
    auto v_order = pattern::any_input();
    
    // KV-path may already have Transposes that will be rewritten based on PA KV inputs required layout
    auto k_shaped_transposed = pattern::wrap_type<v1::Transpose>({std::make_shared<pattern::op::Or>(OutputVector{k_concat, k_shaped}), k_order});
    auto v_shaped_transposed = pattern::wrap_type<v1::Transpose>({std::make_shared<pattern::op::Or>(OutputVector{v_concat, v_shaped}), v_order});

    // Optional pattern to capture alibi slopes (based on pattern from bloom)
    auto alibi = pattern::any_input();
    auto sdpa_mask = pattern::wrap_type<v1::Multiply>({pattern::any_input(), alibi});
    sdpa_mask = pattern::wrap_type<v1::Reshape>({sdpa_mask, pattern::any_input()});
    sdpa_mask = pattern::wrap_type<v1::Reshape>({sdpa_mask, pattern::any_input()});
    sdpa_mask = pattern::wrap_type<v1::Select>({pattern::any_input(), pattern::any_input(), sdpa_mask});

    auto q = pattern::any_input();
    auto sdpa = pattern::wrap_type<v13::ScaledDotProductAttention>({
        q,
        std::make_shared<pattern::op::Or>(OutputVector{k_concat, k_shaped, k_shaped_transposed, k_simply_shaped}),
        std::make_shared<pattern::op::Or>(OutputVector{v_concat, v_shaped, v_shaped_transposed, v_simply_shaped}),
        std::make_shared<pattern::op::Or>(OutputVector{sdpa_mask, pattern::any_input()})
    });

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS, &kv_parameters, &model_remaining_params, &sliding_window, &parameters_to_remove, &assignes_to_remove](ov::pass::pattern::Matcher& m) {
        std::cout << "____" << matcher_name << "___Matched___" << std::endl;
        OPENVINO_ASSERT(m.get_pattern_value_map().find(sdpa) != m.get_pattern_value_map().end());
        const auto& pattern_map = m.get_pattern_map();
        OPENVINO_ASSERT(pattern_map.find(sdpa) != pattern_map.end());
        auto real_q = pattern_map.at(q);
        // Why do have double check here?

        // takes option that has 4D instead of fine-grained Reshape analysis
        // it avoids complication in the pattern, but we don't really have many options
        auto take_4d = [OV_CAPTURE_CPY_AND_THIS](std::shared_ptr<Node> option1, std::shared_ptr<Node> option2, std::shared_ptr<Node> option3) {
            // Question: should it be get_partial_shape() of output or input ???
            if (pattern_map.find(option1) != pattern_map.end() && pattern_map.at(option1)->input(0).get_partial_shape().rank().get_length() == 4) {
                return pattern_map.at(option1);
            } else if (pattern_map.at(option2)->input(0).get_partial_shape().rank().get_length() == 4) {
                return pattern_map.at(option2);
            } else {
                return pattern_map.at(option3);
            }
        };

        auto real_k = take_4d(k_current, k_current_reshaped, k_current2);
        auto real_v = take_4d(v_current, v_current_reshaped, v_current2);

        // is_cpu() check required here.
        // kv_cache_type required here (CURRENTLY USING STUB)
        auto kv_cache_type = element::i64;
        auto k_parameter = std::make_shared<v0::Parameter>(kv_cache_type, PartialShape{-1, -1, -1, -1});
        auto v_parameter = std::make_shared<v0::Parameter>(kv_cache_type, PartialShape{-1, -1, -1, -1});
        kv_parameters.push_back(k_parameter);
        kv_parameters.push_back(v_parameter);
        auto kv_transpose_order = v0::Constant::create(element::i32, Shape{} ,{0, 2, 1, 3}); //correct Constant creation? What Shape to use?
        auto q_transpose = std::make_shared<v1::Transpose>(real_q, kv_transpose_order);
        auto q_reshape = std::make_shared<v1::Reshape>(q_transpose, v0::Constant::create(element::i32, Shape{1}, {0, 0, -1}), true); //correct Constant creation? What Shape to use?

        std::shared_ptr<Node> k_transpose_order = kv_transpose_order; // eeeh, is it a right way to assign Constants? Maybe I should clone somehow?
        if (pattern_map.find(k_order) != pattern_map.end()) { // reapply transpose found in the graph by manipulating of indices of our Transpose
            k_transpose_order = std::make_shared<v1::Gather>(pattern_map.at(v_order), kv_transpose_order, v0::Constant::create(element::i32, Shape{}, {0}));
        }
        auto k_transpose = std::make_shared<v1::Transpose>(real_k, k_transpose_order);
        auto k_reshape = std::make_shared<v1::Reshape>(k_transpose, v0::Constant::create(element::i32, Shape{1}, {0, 0, -1}), true);

        std::shared_ptr<Node> v_transpose_order = kv_transpose_order; // eeeh, is it a right way to assign Constants? Maybe I should clone somehow?
        if (pattern_map.find(v_order) != pattern_map.end()) { // reapply transpose found in the graph by manipulating of indices of our Transpose
            v_transpose_order = std::make_shared<v1::Gather>(pattern_map.at(v_order), kv_transpose_order, v0::Constant::create(element::i32, Shape{}, {0}));
        }
        auto v_transpose = std::make_shared<v1::Transpose>(real_v, v_transpose_order);
        auto v_reshape = std::make_shared<v1::Reshape>(v_transpose, v0::Constant::create(element::i32, Shape{1}, {0, 0, -1}), true);

        // TODO: Detect whether SDPA in the model graph has `scale` argument set and use it instead of the computed scale below
        // Most likely `scale` will always be a constant in real inference, but dynamic dimension propagation may not always derive it as a constant
        // That's why a sub-graph computing `scale` is built instead of just a constant node.
        auto hidden_shape = std::make_shared<v0::ShapeOf>(real_q);
        auto hidden_dim = std::make_shared<v1::Gather>(hidden_shape, v0::Constant::create(element::i32, Shape{}, {-1}), v0::Constant::create(element::i32, Shape{}, {0}));
        
        auto scale = std::make_shared<v1::Divide>(v0::Constant::create(element::f32, Shape{}, {1}), std::make_shared<v0::Sqrt>(std::make_shared<v0::Convert>(hidden_dim, element::f32)));

        std::shared_ptr<Node> alibi_slopes;
        if (pattern_map.find(alibi) != pattern_map.end()) {
            std::cout << "alibi slopes applied" << std::endl;
            alibi_slopes = std::make_shared<v1::Reshape>(pattern_map.at(alibi), v0::Constant::create(element::i32, Shape{1}, {-1}), false); //here {-1} is interesting in Python TODO: discuss
            if (alibi_slopes->get_element_type() == element::f32) {
                alibi_slopes = std::make_shared<v0::Convert>(alibi_slopes, element::f32); //todo
            }
        } else {
            alibi_slopes = v0::Constant::create(element::f32, Shape{1}, {}); // correctly created?
        }

        OutputVector params = {
            q_reshape,
            k_reshape,
            v_reshape,
            k_parameter,
            v_parameter
        };
        params.insert(params.end(), model_remaining_params.begin(), model_remaining_params.end());
        std::initializer_list<std::shared_ptr<Node>> additional_params = {scale, alibi_slopes, sliding_window};
        params.insert(params.end(), additional_params.begin(), additional_params.end());

        // Really not sure if I construct correctly because the Python code uses an additional function
        auto paged_attention = std::make_shared<ov::op::PagedAttentionExtension>(params);

        auto pa_shape = std::make_shared<v0::Concat>(OutputVector{
            v0::Constant::create(element::i32, Shape{1}, {0}),
            v0::Constant::create(element::i32, Shape{1}, {0}),
            v0::Constant::create(element::i32, Shape{1}, {-1}),
            std::make_shared<v0::Unsqueeze>(hidden_dim, v0::Constant::create(element::i32, Shape{}, {0})),
        }, 0);
        auto pa_reshape = std::make_shared<v1::Reshape>(paged_attention, pa_shape, true);
        auto pa_transpose = std::make_shared<v1::Transpose>(pa_reshape, kv_transpose_order);

        // TODO: Complete this part to work with stateless models as well as will stateful
        //  def add_kv_parameter(past_node):
        //      if past_node.get_type_info().name == 'Parameter':
        //          parameters_to_remove.append(past_node)

        //  add_kv_parameter(mapping[k_gather])
        //  add_kv_parameter(mapping[v_gather])

        if (pattern_map.find(v_past_par) != pattern_map.end()) {
            auto param = std::dynamic_pointer_cast<v0::Parameter>(pattern_map.at(v_past_par));
            if (param) {
                return false;
            }
            parameters_to_remove.push_back(param);
        }

        if (pattern_map.find(k_past_par) != pattern_map.end()) {
            auto param = std::dynamic_pointer_cast<v0::Parameter>(pattern_map.at(k_past_par));
            if (param) {
                return false;
            }
            parameters_to_remove.push_back(param);
        }

        auto add_assign_consumers = [OV_CAPTURE_CPY_AND_THIS, &assignes_to_remove](const std::shared_ptr<ov::Output<Node>>& output) {
            for (auto& consumer : output->get_target_inputs()) {
                auto consumer_node = consumer.get_node();
                auto consumer_type = consumer_node->get_type_info().name;
                if (std::strcmp(consumer_type, "Assign") == 0) { // stateful model
                    assignes_to_remove.push_back(std::shared_ptr<Node>(consumer_node));
                } else if (std::strcmp(consumer_type, "Result") == 0) { // stateless model
                    assignes_to_remove.push_back(std::shared_ptr<Node>(consumer_node));
                }
            }
        };

        add_assign_consumers(std::make_shared<ov::Output<Node>>(pattern_map.at(k_concat)->output(0)));
        add_assign_consumers(std::make_shared<ov::Output<Node>>(pattern_map.at(v_concat)->output(0)));

        replace_node(m.get_match_root(), pa_transpose);
        std::cout << "INSERTED PageAttentionExtension" << std::endl;
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sdpa, matcher_name);
    register_matcher(m, callback);
}