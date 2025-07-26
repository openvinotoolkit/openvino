// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "transformations/common_optimizations/pack_GQA.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/block_collection.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace ov::op;
using namespace ov::pass;
using namespace ov::pass::pattern;

namespace {

struct Weights {
    std::shared_ptr<Node> weights;
    std::shared_ptr<Node> scale;
    std::shared_ptr<Node> zero_point;
    std::shared_ptr<Node> mul;

    explicit Weights(const std::shared_ptr<Node>& w) : weights(w) {}
    Weights(const std::shared_ptr<Node>& w,
            const std::shared_ptr<Node>& s,
            const std::shared_ptr<Node>& zp,
            const std::shared_ptr<Node>& mul)
        : weights(w),
          scale(s),
          zero_point(zp),
          mul(mul) {}

    bool is_quantized() const {
        return scale && zero_point;
    }

    Output<Node> get_data() const {
        return weights->output(0);
    }
};

std::tuple<ov::NodeVector, Node*> get_sdpa_order(const std::unordered_set<Node*>& post_sdpa_proj) {
    ov::NodeVector post_sdpa_ordered;
    ov::op::v1::Add* current_add = nullptr;

    for (const auto& proj_node : post_sdpa_proj) {
        const auto& targets = proj_node->output(0).get_target_inputs();
        if (targets.size() != 1)
            continue;

        auto input = *targets.begin();
        auto add_node = ov::as_type<ov::op::v1::Add>(input.get_node());
        if (!add_node)
            continue;

        auto lhs = add_node->input_value(0).get_node();
        auto rhs = add_node->input_value(1).get_node();
        if (post_sdpa_proj.count(lhs) && post_sdpa_proj.count(rhs)) {
            current_add = add_node;
            post_sdpa_ordered.push_back(lhs->shared_from_this());
            post_sdpa_ordered.push_back(rhs->shared_from_this());
            break;
        }
    }

    if (!current_add)
        return {};

    while (true) {
        const auto& targets = current_add->output(0).get_target_inputs();
        if (targets.size() != 1)
            return {};

        auto next_node_input = targets.begin();
        current_add = ov::as_type<ov::op::v1::Add>(next_node_input->get_node());
        if (!current_add)
            break;

        auto another_idx = 1 - next_node_input->get_index();
        auto input_node = current_add->input_value(another_idx).get_node();
        if (post_sdpa_proj.count(input_node)) {
            post_sdpa_ordered.push_back(input_node->shared_from_this());
        } else {
            break;
        }
    }
    return {post_sdpa_ordered, current_add};
}

Output<Node> fuse_weights_and_replace(const std::vector<Weights>& weights_list, int64_t concat_dim_idx) {
    auto ensure_3d = [](const Output<Node>& input) -> Output<Node> {
        if (input.get_partial_shape().rank().is_static() && input.get_partial_shape().rank().get_length() == 2) {
            auto shape = v0::Constant::create(element::i64, Shape{3}, {0, -1, 1});
            return ov::op::util::make_try_fold<v1::Reshape>(input, shape, true);
        }
        return input;
    };

    if (weights_list.empty())
        return {};

    const auto& ref = weights_list.front();
    auto weights_consumers = ref.weights->output(0).get_target_inputs();

    // Concatenate weights
    OutputVector weights_values;
    for (const auto& w : weights_list)
        weights_values.push_back(ensure_3d(w.weights));

    auto fused_weights = ov::op::util::make_try_fold<v0::Concat>(weights_values, 2);

    // Replace only the weights node
    for (auto& input : weights_consumers)
        input.replace_source_output(fused_weights);

    // If quantized: also replace scale and zero point
    if (ref.is_quantized()) {
        OutputVector scales;
        OutputVector zero_points;

        auto scale_consumers = ref.scale->output(0).get_target_inputs();
        auto zp_consumers = ref.zero_point->output(0).get_target_inputs();
        for (const auto& w : weights_list) {
            auto ensure_1d = [](const Output<Node>& input) -> Output<Node> {
                if (input.get_partial_shape().rank().is_static()) {
                    auto shape = v0::Constant::create(element::i64, Shape{3}, {1, 1, 1});
                    return ov::op::util::make_try_fold<v1::Reshape>(input, shape, true);
                }
                return input;
            };
            scales.push_back(ensure_1d(w.scale));
            zero_points.push_back(ensure_1d(w.zero_point));
        }

        auto fused_scale = ov::op::util::make_try_fold<v0::Concat>(scales, 2);
        auto fused_zp = ov::op::util::make_try_fold<v0::Concat>(zero_points, 2);

        for (auto& input : scale_consumers)
            input.replace_source_output(fused_scale);

        for (auto& input : zp_consumers)
            input.replace_source_output(fused_zp);

        auto shape = v0::Constant::create(element::i64, Shape{2}, {0, -1});
        auto mul_reshaped = std::make_shared<v1::Reshape>(ref.mul, shape, true);
        ov::replace_node(ref.mul, mul_reshaped);
    }

    return {};
}

}  // namespace

PackGQA::PackGQA() : MultiMatcher("PackGQA") {
    // Pattern 1
    auto norm_input = any_input();
    auto norm_block = blocks::l2_norm_block(norm_input);

    auto weights_constant = wrap_type<v0::Constant>();
    auto opt_convert = optional<v0::Convert>(weights_constant);
    auto proj_dq = blocks::dq_constant_block();
    auto qkv_projections = wrap_type<v0::MatMul>({norm_block, proj_dq | opt_convert});

    auto bias_const = wrap_type<v0::Constant>();
    auto opt_bias_convert = optional<v0::Convert>(bias_const);
    auto proj_bias = wrap_type<v1::Add>({qkv_projections, opt_bias_convert});

    // Pattern 2
    auto q_input = any_input();
    auto k_input = any_input();
    auto v_input = any_input();

    auto q = blocks::sdpa_preprocessing_block(q_input);
    auto k = blocks::sdpa_preprocessing_block(k_input);

    auto reshape_v = wrap_type<v1::Reshape>({v_input, any_input()});
    auto vT = optional<v1::Transpose>({reshape_v, any_input()});

    auto sdpa = blocks::sdpa_block(q, k, vT);
    auto t2 = wrap_type<v1::Transpose>({sdpa, any_input()});
    auto reshaped = wrap_type<v1::Reshape>({t2, any_input()});

    auto lin_weights_constant = wrap_type<v0::Constant>();
    auto lin_opt_convert = optional<v0::Convert>(lin_weights_constant);

    auto lin_proj_dq = blocks::dq_constant_block();
    auto proj = wrap_type<v0::MatMul>({reshaped, lin_proj_dq | lin_opt_convert});

    auto callback = [=](const std::unordered_map<std::shared_ptr<Node>, std::vector<PatternValueMap>>& matches) {
        if (matches.size() != 2) {
            return;
        }

        std::unordered_set<Node*> post_sdpa_proj;
        std::unordered_map<Node*, const PatternValueMap*> node_to_proj_pm;
        for (const auto& pm : matches.at(proj)) {
            auto root = pm.at(proj).get_node();
            post_sdpa_proj.insert(root);
            node_to_proj_pm[root] = &pm;
        }

        std::unordered_map<Node*, const PatternValueMap*> node_to_bias_pm;
        for (const auto& pm : matches.at(proj_bias)) {
            auto root = pm.at(proj_bias).get_node_shared_ptr();
            node_to_bias_pm[root.get()] = &pm;
        }

        auto [post_sdpa_proj_ordered, node_after_mha] = get_sdpa_order(post_sdpa_proj);
        if (post_sdpa_proj_ordered.empty())
            return;

        std::vector<Weights> q_weights, k_weights, v_weights;
        std::vector<Weights> q_biases, k_biases, v_biases;
        std::vector<Weights> linear_projection;

        for (const auto& node : post_sdpa_proj_ordered) {
            const auto* pm = node_to_proj_pm.at(node.get());

            if (pm->count(lin_proj_dq)) {
                auto block =
                    std::dynamic_pointer_cast<ov::pass::pattern::op::Block>(pm->at(lin_proj_dq).get_node_shared_ptr());

                linear_projection.emplace_back(block->get_anchor("constant", *pm).value().get_node_shared_ptr(),
                                               block->get_anchor("scale", *pm).has_value()
                                                   ? block->get_anchor("scale", *pm).value().get_node_shared_ptr()
                                                   : nullptr,
                                               block->get_anchor("zp", *pm).has_value()
                                                   ? block->get_anchor("zp", *pm).value().get_node_shared_ptr()
                                                   : nullptr,
                                               block->get_anchor("mul", *pm).has_value()
                                                   ? block->get_anchor("mul", *pm).value().get_node_shared_ptr()
                                                   : nullptr);
            } else {
                linear_projection.emplace_back(pm->at(lin_weights_constant).get_node_shared_ptr());
            }

            auto process =
                [&](const Output<Node>& input, std::vector<Weights>& weights_vec, std::vector<Weights>& bias_vec) {
                    auto input_node = pm->at(input.get_node_shared_ptr()).get_node_shared_ptr();
                    const auto* bias_pm = node_to_bias_pm.at(input_node.get());

                    if (bias_pm->count(proj_dq)) {
                        auto dq = std::dynamic_pointer_cast<ov::pass::pattern::op::Block>(
                            bias_pm->at(proj_dq).get_node_shared_ptr());
                        weights_vec.emplace_back(dq->get_anchor("constant", *bias_pm).value().get_node_shared_ptr(),
                                                 dq->get_anchor("scale", *bias_pm).has_value()
                                                     ? dq->get_anchor("scale", *bias_pm).value().get_node_shared_ptr()
                                                     : nullptr,
                                                 dq->get_anchor("zp", *bias_pm).has_value()
                                                     ? dq->get_anchor("zp", *bias_pm).value().get_node_shared_ptr()
                                                     : nullptr,
                                                 dq->get_anchor("mul", *bias_pm).has_value()
                                                     ? dq->get_anchor("mul", *bias_pm).value().get_node_shared_ptr()
                                                     : nullptr);
                    } else if (bias_pm->count(weights_constant)) {
                        weights_vec.emplace_back(bias_pm->at(weights_constant).get_node_shared_ptr());
                    }

                    if (bias_pm->count(opt_bias_convert)) {
                        bias_vec.emplace_back(bias_pm->at(opt_bias_convert).get_node_shared_ptr());
                    } else if (bias_pm->count(bias_const)) {
                        bias_vec.emplace_back(bias_pm->at(bias_const).get_node_shared_ptr());
                    }
                };

            process(q_input, q_weights, q_biases);
            process(k_input, k_weights, k_biases);
            process(v_input, v_weights, v_biases);
        }

        for (const auto& weights : {q_weights, k_weights, v_weights, linear_projection}) {
            fuse_weights_and_replace(weights, 1);
        }

        for (const auto& biases : {q_biases, k_biases, v_biases}) {
            fuse_weights_and_replace(biases, 0);
        }

        const auto* proj_pm = node_to_proj_pm.at(post_sdpa_proj_ordered[0].get());
        auto proj_transpose = proj_pm->at(t2).get_node_shared_ptr();

        auto axis_0 = v0::Constant::create(element::i64, Shape{1}, {2});
        auto reduce_0 = std::make_shared<v1::ReduceSum>(proj_transpose, axis_0, false);

        auto proj_reshape = proj_pm->at(reshaped).get_node_shared_ptr();
        proj_reshape->input(0).replace_source_output(reduce_0->output(0));
        auto proj_matmul = proj_pm->at(proj).get_node_shared_ptr();

        int head_size = post_sdpa_proj_ordered.size();
        auto reshape_shape = v0::Constant::create(element::i64, Shape{4}, {0, 0, -1, head_size});
        auto reshaped = std::make_shared<v1::Reshape>();
        reshaped->set_argument(0, proj_matmul->output(0));
        reshaped->set_argument(1, reshape_shape);
        reshaped->set_special_zero(true);

        auto axis = v0::Constant::create(element::i64, Shape{1}, {3});
        auto reduced = std::make_shared<v1::ReduceSum>(reshaped, axis, false);
        node_after_mha->input(0).replace_source_output(reduced);
    };

    register_patterns({proj_bias, proj}, callback, true);
}