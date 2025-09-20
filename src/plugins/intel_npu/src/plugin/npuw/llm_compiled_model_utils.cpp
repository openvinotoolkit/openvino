// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_compiled_model_utils.hpp"

#include "logging.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/validate.hpp"

namespace opp = ov::pass::pattern;

// diagnostics warnings on OPENVINO_MATCHER_PASS_RTTI() definition: visibility hidden
#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wattributes"
#endif
namespace {

class TransposeValueTensors : public ov::pass::MatcherPass {
public:
    struct Context {
        using Ref = std::reference_wrapper<Context>;
        bool bTransposed = false;
    };

protected:
    // generic part of matchers, to transpose v-tensors, and concat, and update matmul args
    void transpose_matmul_b(Context::Ref ctx,
                            const std::shared_ptr<ov::op::v0::Parameter>& matched_param,
                            const std::shared_ptr<ov::op::v0::Concat>& matched_concat,
                            const std::shared_ptr<ov::op::v1::Transpose>& matched_transpose,
                            const std::shared_ptr<ov::op::v0::MatMul>& matched_matmul) {
        auto param_shape = matched_param->get_partial_shape();
        NPUW_ASSERT(param_shape.size() == 4u);
        // NB: Transpose Parameter that correspond to V-tensor it will
        // speed-up its multiplication with attention scores
        std::swap(param_shape[2], param_shape[3]);

        matched_param->set_partial_shape(param_shape);

        auto order_cst = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 2, 3, 1});

        matched_transpose->set_argument(1, order_cst);
        matched_concat->set_axis(3u);
        matched_matmul->set_transpose_b(true);
        ctx.get().bTransposed = true;
    }

    void transpose_matmul_b(Context::Ref ctx,
                            const std::shared_ptr<ov::Node>& node_param,
                            const std::shared_ptr<ov::Node>& node_concat,
                            const std::shared_ptr<ov::Node>& node_transpose,
                            const std::shared_ptr<ov::Node>& node_matmul) {
        auto matched_param = std::static_pointer_cast<ov::op::v0::Parameter>(node_param);
        auto matched_concat = std::static_pointer_cast<ov::op::v0::Concat>(node_concat);
        auto matched_transpose = std::static_pointer_cast<ov::op::v1::Transpose>(node_transpose);
        auto matched_matmul = std::static_pointer_cast<ov::op::v0::MatMul>(node_matmul);

        transpose_matmul_b(ctx, matched_param, matched_concat, matched_transpose, matched_matmul);
    }
};

// llama2 pattern for value tensor concate
class TransposeValueTensors_llama2 : public TransposeValueTensors {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::TransposeValueTensors_llama2");
    TransposeValueTensors_llama2(Context::Ref ctx) {
        register_matcher_llama2(ctx);
    }

private:
    void register_matcher_llama2(Context::Ref ctx) {
        auto param = opp::wrap_type<ov::op::v0::Parameter>();
        auto transpose = opp::wrap_type<ov::op::v1::Transpose>({opp::any_input(), opp::any_input()});
        auto convert = opp::optional<ov::op::v0::Convert>({param->output(0)});
        auto concat = opp::wrap_type<ov::op::v0::Concat>({convert, transpose});
        auto softmax = opp::wrap_type<ov::op::v8::Softmax>({opp::any_input()});
        auto matmul = opp::wrap_type<ov::op::v0::MatMul>({softmax, concat});

        auto callback = [=](ov::pass::pattern::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();

            auto matched_node_param = node_to_output.at(param).get_node_shared_ptr();
            auto matched_node_concat = node_to_output.at(concat).get_node_shared_ptr();
            auto matched_node_transpose = node_to_output.at(transpose).get_node_shared_ptr();
            auto matched_node_matmul = node_to_output.at(matmul).get_node_shared_ptr();

            transpose_matmul_b(ctx,
                               matched_node_param,
                               matched_node_concat,
                               matched_node_transpose,
                               matched_node_matmul);
            LOG_DEBUG("vtensors transposed: LLama2 pattern");
            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(matmul, "TransposeValueTensors_llama2"), std::move(callback));
    }
};

// llama3, phi3, mistral, etc, concate value tensors with broadcasting
class TransposeValueTensors_llama3 : public TransposeValueTensors {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::TransposeValueTensors_llama3");
    TransposeValueTensors_llama3(Context::Ref ctx) {
        register_matcher_llama3(ctx);
    }

private:
    void register_matcher_llama3(Context::Ref ctx) {
        auto param = opp::wrap_type<ov::op::v0::Parameter>();
        auto transpose = opp::wrap_type<ov::op::v1::Transpose>({opp::any_input(), opp::any_input()});
        auto convert = opp::optional<ov::op::v0::Convert>({param->output(0)});
        auto concat = opp::wrap_type<ov::op::v0::Concat>({convert, transpose});

        // only difference is that broadcast wrapped into unsquese/reshape, while transposed tensor didn't change
        const auto unsqueeze_axes = opp::wrap_type<ov::op::v0::Constant>();
        auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({concat, unsqueeze_axes});
        auto broadcast = opp::wrap_type<ov::op::v1::Broadcast, ov::op::v3::Broadcast>({unsqueeze, opp::any_input()});
        auto reshape = opp::wrap_type<ov::op::v1::Reshape>({broadcast, opp::any_input()});

        // v8 softmax? what? can be other softmaxes
        auto softmax = opp::wrap_type<ov::op::v8::Softmax>({opp::any_input()});
        auto matmul = opp::wrap_type<ov::op::v0::MatMul>({softmax, reshape});

        auto callback = [=](ov::pass::pattern::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();

            auto matched_node_param = node_to_output.at(param).get_node_shared_ptr();
            auto matched_node_concat = node_to_output.at(concat).get_node_shared_ptr();
            auto matched_node_transpose = node_to_output.at(transpose).get_node_shared_ptr();
            auto matched_node_matmul = node_to_output.at(matmul).get_node_shared_ptr();
            auto matched_node_unsqueeze = node_to_output.at(unsqueeze).get_node_shared_ptr();
            auto matched_node_unsqueeze_axes = node_to_output.at(unsqueeze_axes).get_node_shared_ptr();
            auto matched_node_broadcast = node_to_output.at(broadcast).get_node_shared_ptr();
            auto matched_node_reshape = node_to_output.at(reshape).get_node_shared_ptr();

            auto matched_param = std::static_pointer_cast<ov::op::v0::Parameter>(matched_node_param);
            auto matched_concat = std::static_pointer_cast<ov::op::v0::Concat>(matched_node_concat);
            auto matched_transpose = std::static_pointer_cast<ov::op::v1::Transpose>(matched_node_transpose);
            auto matched_matmul = std::static_pointer_cast<ov::op::v0::MatMul>(matched_node_matmul);
            auto matched_unsqueeze = std::static_pointer_cast<ov::op::v0::Unsqueeze>(matched_node_unsqueeze);
            auto matched_broadcast = std::static_pointer_cast<ov::op::v3::Broadcast>(matched_node_broadcast);
            auto matched_reshape = std::static_pointer_cast<ov::op::v1::Reshape>(matched_node_reshape);

            auto shape_broadcast = matched_broadcast->get_output_shape(0);
            NPUW_ASSERT(shape_broadcast.size() == 5u);
            std::swap(shape_broadcast[3], shape_broadcast[4]);

            LOG_DEBUG("shape_broadcast for: " << matched_broadcast->get_friendly_name()
                                              << ", shape=" << shape_broadcast);

            const auto broadcast_axes_node =
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{5}, shape_broadcast);
            broadcast_axes_node->set_friendly_name(matched_broadcast->get_friendly_name() + "/new_broadcast_shape");
            matched_broadcast->input(1).replace_source_output(broadcast_axes_node);

            auto shape_reshape = matched_reshape->get_output_shape(0);
            NPUW_ASSERT(shape_reshape.size() == 4u);
            std::swap(shape_reshape[2], shape_reshape[3]);

            LOG_DEBUG("shape_reshape for: " << matched_reshape->get_friendly_name() << ", shape=" << shape_reshape);

            const auto reshape_axes_node =
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, shape_reshape);
            reshape_axes_node->set_friendly_name(matched_reshape->get_friendly_name() + "/new_reshape_shape");
            matched_reshape->input(1).replace_source_output(reshape_axes_node);

            transpose_matmul_b(ctx, matched_param, matched_concat, matched_transpose, matched_matmul);
            LOG_DEBUG("vtensors transposed: LLama3 pattern");
            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(matmul, "TransposeValueTensors_llama3"), std::move(callback));
    }
};

class ScaledDotProductAttentionDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::LLMCompiledModel::ScaledDotProductAttentionDecomposition");
    explicit ScaledDotProductAttentionDecomposition(bool use_high_precision_on_add) {
        auto pattern_node = ov::pass::pattern::wrap_type<ov::op::v13::ScaledDotProductAttention>();

        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            auto& pattern_to_output = m.get_pattern_value_map();
            auto node = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(
                pattern_to_output.at(pattern_node).get_node_shared_ptr());

            if (node == nullptr || transformation_callback(node)) {
                return false;
            }

            auto new_output_node = decompose(node, use_high_precision_on_add);
            ov::replace_node(node, new_output_node);
            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern_node, "ScaledDotProductAttentionDecomposition");
        register_matcher(m, std::move(callback));
    }
    std::shared_ptr<ov::Node> decompose(std::shared_ptr<ov::op::v13::ScaledDotProductAttention> node,
                                        bool use_high_precision_on_add) {
        using namespace ov::op;
        using namespace ov;
        auto query = node->input_value(0);
        auto key = node->input_value(1);
        auto value = node->input_value(2);
        auto q_shape = register_new_node<v3::ShapeOf>(query, element::i32);
        auto k_shape = register_new_node<v3::ShapeOf>(key, element::i32);
        auto minus_one = register_new_node(v0::Constant::create(element::i32, Shape{}, {-1}));
        auto minus_two = register_new_node(v0::Constant::create(element::i32, Shape{}, {-2}));
        auto zero_i = register_new_node(v0::Constant::create(element::i32, Shape{}, {0}));
        auto one_i = register_new_node(v0::Constant::create(element::i32, Shape{}, {1}));
        auto one_f = register_new_node<v1::ConvertLike>(one_i, query);
        auto zero_f = register_new_node<v1::ConvertLike>(zero_i, query);

        Output<Node> scale;
        if (node->get_input_size() < 5) {
            scale = register_new_node<v8::Gather>(q_shape, minus_one, zero_i)->output(0);
            scale = register_new_node<v1::ConvertLike>(scale, query);
            auto sqrt_scale = register_new_node<v0::Sqrt>(scale);
            scale = register_new_node<v1::Divide>(one_f, sqrt_scale);
        } else {
            scale = node->input_value(4);
        }

        auto q_scaled = register_new_node<v1::Multiply>(query, scale);
        auto k_rank = register_new_node<v3::ShapeOf>(k_shape, element::i32)->output(0);
        auto k_last_dim = register_new_node<v1::Add>(k_rank, minus_one);
        auto k_next_dim = register_new_node<v1::Add>(k_rank, minus_two)->output(0);
        k_rank = register_new_node<v0::Squeeze>(k_rank, zero_i);
        auto minus_inf =
            register_new_node(v0::Constant::create(element::f32, Shape{}, {-std::numeric_limits<float>::infinity()}))
                ->output(0);
        auto keep_dim_last = register_new_node<v0::Squeeze>(k_next_dim, zero_i);
        auto k_dims_before_transpose = register_new_node<v4::Range>(zero_i, keep_dim_last, one_i, element::i32);

        auto scaled_atten = register_new_node<v0::MatMul>(q_scaled, key, false, true)->output(0);
        minus_inf = register_new_node<v1::ConvertLike>(minus_inf, scaled_atten);

        if (node->get_causal() || node->get_input_size() > 3) {
            Output<Node> mask;
            Output<Node> atten_mask;
            if (!node->get_causal()) {
                mask = node->input_value(3);

                // two types of masks are supported. A boolean mask where a value of True indicates that the element
                // should take part in attention. A float mask of the same type as query, key, value that is added to
                // the attention score.
                if (mask.get_element_type() == element::boolean) {
                    atten_mask = register_new_node<v1::Select>(mask, zero_f, minus_inf);
                } else {
                    atten_mask = mask;
                }
            } else {
                auto target_s_len = register_new_node<v8::Gather>(q_shape, minus_two, zero_i);
                auto source_s_len = register_new_node<v8::Gather>(k_shape, minus_two, zero_i);
                auto ssl = register_new_node<v0::Unsqueeze>(source_s_len, zero_i);
                auto tsl = register_new_node<v0::Unsqueeze>(target_s_len, zero_i);
                auto mask_shape = register_new_node<v0::Concat>(OutputVector{tsl, ssl}, 0);
                mask = register_new_node<v1::Broadcast>(minus_inf, mask_shape);
                auto horizontal_range =
                    register_new_node<v4::Range>(zero_i, source_s_len, one_i, element::i32)->output(0);
                horizontal_range = register_new_node<v0::Unsqueeze>(horizontal_range, zero_i);
                auto stop = register_new_node<v1::Add>(target_s_len, one_i);
                auto vertical_range = register_new_node<v4::Range>(one_i, stop, one_i, element::i32)->output(0);
                vertical_range = register_new_node<v0::Unsqueeze>(vertical_range, one_i);
                auto triu = register_new_node<v1::GreaterEqual>(horizontal_range, vertical_range);
                atten_mask = register_new_node<v1::Select>(triu, mask, zero_f);
            }
            if (use_high_precision_on_add) {
                npuw::util::HighPrecisionAttr attr_hp;
                attr_hp.compute_precision_type = ov::element::f32;
                atten_mask.get_rt_info()[npuw::util::HighPrecisionAttr::get_type_info_static()] = attr_hp;
            }

            scaled_atten = register_new_node<v1::Add>(scaled_atten, atten_mask);
        }

        scaled_atten = register_new_node<v8::Softmax>(scaled_atten, -1);
        auto result = register_new_node<v0::MatMul>(scaled_atten, value);
        result->set_friendly_name(node->get_friendly_name());
        copy_runtime_info(node, get_new_nodes());
        return result;
    }
};

}  // namespace

#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif

namespace ov::npuw::util {
bool optimize_value_tensors(std::shared_ptr<ov::Model> model, bool isPrefill) {
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ScaledDotProductAttentionDecomposition>(isPrefill);
    TransposeValueTensors::Context ctx;
    rewr.add_matcher<TransposeValueTensors_llama2>(std::ref(ctx));
    rewr.add_matcher<TransposeValueTensors_llama3>(std::ref(ctx));
    rewr.run_on_model(model);

    ov::pass::Validate().run_on_model(model);

    // NB: matmul parameters gets transposed, if pass applied
    return ctx.bTransposed;
}
}  // namespace ov::npuw::util
