// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/activations_scaling.hpp"

#include <memory>

#include "itt.hpp"
#include "low_precision/network_helper.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/group_normalization.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rms.hpp"
#include "transformations/common_optimizations/lin_op_sequence_fusion.hpp"
#include "transformations/utils/utils.hpp"

namespace {
const auto is_scalar_node = [](const ov::Output<ov::Node>& output) -> bool {
    const auto shape = output.get_partial_shape();
    if (shape.is_dynamic())
        return false;
    return ov::shape_size(shape.to_shape()) == 1;
};

const auto is_non_const_node = [](const ov::Output<ov::Node>& output) -> bool {
    return !ov::is_type<ov::op::v0::Constant>(output.get_node());
};
}  // namespace

using namespace ov::pass::activations_scaling;
using namespace ov::pass::pattern;
using ov::pass::pattern::op::Or;

ov::pass::activations_scaling::ScaleDownSingleLayer::ScaleDownSingleLayer(float scale_factor,
                                                                          ov::element::Type scaled_prec) {
    MATCHER_SCOPE(ScaleDownSingleLayer);

    auto activation_m = any_input();
    auto weights_m = any_input();
    auto convolution_m = wrap_type<ov::op::v1::Convolution>({activation_m, weights_m});
    auto matmul_m = wrap_type<ov::op::v0::MatMul>({activation_m, weights_m});
    auto scaled_op_m = std::make_shared<Or>(OutputVector{convolution_m, matmul_m});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        OPENVINO_ASSERT(pattern_map.count(convolution_m) || pattern_map.count(matmul_m),
                        "Not found any Convolution or MatMul layer");

        auto insert_scale_down_layer = [&scale_factor, &scaled_prec](std::shared_ptr<ov::Node>& node,
                                                                     const size_t input_idx) {
            const std::vector<float> scale_down_value = {1.f / scale_factor};

            auto scale_down_layer = std::make_shared<ov::op::v1::Multiply>(
                node->input(input_idx).get_source_output(),
                std::make_shared<ov::op::v0::Constant>(node->input(input_idx).get_element_type(),
                                                       ov::Shape(),
                                                       scale_down_value));
            scale_down_layer->set_friendly_name(node->get_friendly_name() + "_scale_down");
            ov::copy_runtime_info(node, scale_down_layer);

            if (scale_down_layer->output(0).get_element_type() != scaled_prec) {
                auto convert_prec = std::make_shared<ov::op::v0::Convert>(scale_down_layer->output(0), scaled_prec);
                node->input(input_idx).replace_source_output(convert_prec->output(0));
            } else {
                node->input(input_idx).replace_source_output(scale_down_layer->output(0));
            }
        };

        // scale_down and scale_up layers will be added around scaled_op
        std::shared_ptr<ov::Node> scaled_op = nullptr;

        if (pattern_map.count(convolution_m))
            scaled_op = pattern_map.at(convolution_m).get_node_shared_ptr();

        if (pattern_map.count(matmul_m))
            scaled_op = pattern_map.at(matmul_m).get_node_shared_ptr();

        if (transformation_callback(scaled_op))
            return false;

        // in the case of decompressed_to_f32 nodes, no need to apply activations scaling
        std::shared_ptr<ov::Node> output_of_scaled_op = scaled_op;
        auto child_node = scaled_op->get_output_target_inputs(0).begin()->get_node();
        if (scaled_op->get_output_target_inputs(0).size() == 1 && ov::is_type<ov::op::v0::Convert>(child_node) &&
            ov::fp16_compression_is_disabled(child_node->shared_from_this()) &&
            ov::pass::constant_folding_is_disabled(child_node->shared_from_this())) {
            return false;
        }

        const std::vector<float> scale_up_value = {scale_factor};
        auto output_prec = output_of_scaled_op->output(0).get_element_type();

        // adding a scale_down layer before the target node
        insert_scale_down_layer(scaled_op, 0);
        if (scaled_op->input(1).get_element_type() != scaled_prec) {
            auto convert_prec1 =
                std::make_shared<ov::op::v0::Convert>(scaled_op->input(1).get_source_output(), scaled_prec);
            scaled_op->input(1).replace_source_output(convert_prec1->output(0));
        }

        scaled_op->revalidate_and_infer_types();

        std::set<ov::Input<ov::Node>> target_inputs;

        // If the target node has a bias layer, scale_up layer will be added after the bias layer.
        // So, we need to scale_down the bias layer too.
        bool has_bias = false;
        size_t bias_index = 1;
        {
            if (scaled_op->get_output_target_inputs(0).size() == 1 && ov::is_type<ov::op::v1::Add>(child_node)) {
                bias_index = (child_node->get_input_node_shared_ptr(0) == scaled_op) ? 1 : 0;
                const auto& bias_pshape = child_node->get_input_partial_shape(bias_index);
                if (bias_pshape.is_static()) {
                    const auto& bias_shape = bias_pshape.get_shape();
                    const bool per_channel = std::count_if(bias_shape.begin(), bias_shape.end(), [](size_t x) {
                                                 return x > 1;
                                             }) == 1;
                    if (ov::shape_size(bias_shape) == 1 || per_channel) {
                        has_bias = true;
                    }
                }
            }
        }

        if (has_bias) {
            auto add = child_node->shared_from_this();
            target_inputs = add->get_output_target_inputs(0);
            insert_scale_down_layer(add, bias_index);
            add->revalidate_and_infer_types();
            if (add->output(0).get_element_type() != output_prec) {
                output_of_scaled_op = std::make_shared<ov::op::v0::Convert>(add->output(0), output_prec);
            } else {
                output_of_scaled_op = add;
            }
        } else {
            target_inputs = output_of_scaled_op->get_output_target_inputs(0);
            if (output_of_scaled_op->output(0).get_element_type() != output_prec) {
                output_of_scaled_op =
                    std::make_shared<ov::op::v0::Convert>(output_of_scaled_op->output(0), output_prec);
            }
        }

        auto scale_up = register_new_node<ov::op::v1::Multiply>(
            output_of_scaled_op->output(0),
            std::make_shared<ov::op::v0::Constant>(output_of_scaled_op->output(0).get_element_type(),
                                                   ov::Shape(),
                                                   scale_up_value));
        scale_up->set_friendly_name(scaled_op->get_friendly_name() + "_scale_up");
        ov::copy_runtime_info(scaled_op, scale_up);
        for (auto& in : target_inputs) {
            in.replace_source_output(scale_up);
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(scaled_op_m, "ScaleDownSingleLayer");
    this->register_matcher(m, callback);
}

ov::pass::activations_scaling::EliminateScalarMul::EliminateScalarMul() {
    MATCHER_SCOPE(EliminateScalarMul);

    auto activation_m = any_input(is_non_const_node);
    auto convert_m = ov::pass::pattern::optional<ov::op::v0::Convert>(activation_m);
    auto scale_const_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(is_scalar_node);
    auto mul_m = wrap_type<ov::op::v1::Multiply>({convert_m, scale_const_m});
    auto mvn_m = wrap_type<ov::op::v6::MVN>({mul_m, any_input()});
    auto rms_m = wrap_type<ov::op::internal::RMS>({mul_m, any_input()});
    auto group_norm_m = wrap_type<ov::op::v12::GroupNormalization>({mul_m, any_input(), any_input()});
    auto shape_of_m = wrap_type<ov::op::v3::ShapeOf>({mul_m});
    auto norm_m = std::make_shared<Or>(OutputVector{mvn_m, rms_m, group_norm_m, shape_of_m});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto scale_const = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(scale_const_m).get_node_shared_ptr());

        if (pattern_map.count(shape_of_m) == 0 && scale_const->cast_vector<float>()[0] < 1.f)
            return false;

        auto activation = pattern_map.at(activation_m);
        auto norm = pattern_map.at(norm_m).get_node_shared_ptr();

        norm->input(0).replace_source_output(activation);

        if (pattern_map.count(rms_m)) {
            auto rms = ov::as_type_ptr<ov::op::internal::RMS>(pattern_map.at(rms_m).get_node_shared_ptr());
            auto eps = rms->get_epsilon();
            double scale_const_val = scale_const->cast_vector<double>()[0];
            rms->set_epsilon(eps / scale_const_val / scale_const_val);
        } else if (pattern_map.count(group_norm_m)) {
            auto group_norm =
                ov::as_type_ptr<ov::op::v12::GroupNormalization>(pattern_map.at(group_norm_m).get_node_shared_ptr());
            auto eps = group_norm->get_epsilon();
            double scale_const_val = scale_const->cast_vector<double>()[0];
            group_norm->set_epsilon(eps / scale_const_val / scale_const_val);
        } else if (pattern_map.count(mvn_m)) {
            auto mvn = ov::as_type_ptr<ov::op::v6::MVN>(pattern_map.at(mvn_m).get_node_shared_ptr());
            auto eps_mode = mvn->get_eps_mode();
            auto eps = mvn->get_eps();
            float scale_const_val = scale_const->cast_vector<float>()[0];
            if (eps_mode == ov::op::MVNEpsMode::INSIDE_SQRT) {
                mvn->set_eps(eps / scale_const_val / scale_const_val);
            } else {
                mvn->set_eps(eps / scale_const_val);
            }
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(norm_m, "EliminateScalarMul");
    this->register_matcher(m, callback);
}

ov::pass::activations_scaling::MulShareTransformation::MulShareTransformation() {
    MATCHER_SCOPE(MulShareTransformation);

    auto mvn_m = wrap_type<ov::op::v6::MVN>({any_input(), any_input()});
    auto rms_m = wrap_type<ov::op::internal::RMS>({any_input(), any_input()});
    auto group_norm_m = wrap_type<ov::op::v12::GroupNormalization>({any_input(), any_input(), any_input()});
    auto shape_of_m = wrap_type<ov::op::v3::ShapeOf>({any_input()});
    auto norm_m = std::make_shared<Or>(OutputVector{mvn_m, rms_m, group_norm_m, shape_of_m});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto norm = pattern_map.at(norm_m).get_node_shared_ptr();

        auto parent_output = norm->get_input_source_output(0);
        if (parent_output.get_target_inputs().size() == 1)
            return false;

        for (auto& child : parent_output.get_target_inputs()) {
            if (child == norm->input(0))
                continue;

            if (ov::is_type<ov::op::v1::Multiply>(child.get_node())) {
                ov::Output<ov::Node> const_input;
                for (auto input : child.get_node()->input_values()) {
                    if (input == parent_output)
                        continue;
                    const_input = input;
                }

                if (is_scalar_node(const_input) && !is_non_const_node(const_input)) {
                    norm->input(0).replace_source_output(child.get_node()->output(0));

                    auto scale_const = ov::as_type_ptr<ov::op::v0::Constant>(const_input.get_node_shared_ptr());

                    if (pattern_map.count(rms_m)) {
                        auto rms = ov::as_type_ptr<ov::op::internal::RMS>(pattern_map.at(rms_m).get_node_shared_ptr());
                        auto eps = rms->get_epsilon();
                        double scale_const_val = scale_const->cast_vector<double>()[0];
                        rms->set_epsilon(eps * scale_const_val * scale_const_val);
                    } else if (pattern_map.count(group_norm_m)) {
                        auto group_norm = ov::as_type_ptr<ov::op::v12::GroupNormalization>(
                            pattern_map.at(group_norm_m).get_node_shared_ptr());
                        auto eps = group_norm->get_epsilon();
                        double scale_const_val = scale_const->cast_vector<double>()[0];
                        group_norm->set_epsilon(eps * scale_const_val * scale_const_val);
                    } else if (pattern_map.count(mvn_m)) {
                        auto mvn = ov::as_type_ptr<ov::op::v6::MVN>(pattern_map.at(mvn_m).get_node_shared_ptr());
                        auto eps_mode = mvn->get_eps_mode();
                        auto eps = mvn->get_eps();
                        float scale_const_val = scale_const->cast_vector<float>()[0];
                        if (eps_mode == ov::op::MVNEpsMode::INSIDE_SQRT) {
                            mvn->set_eps(eps * scale_const_val * scale_const_val);
                        } else {
                            mvn->set_eps(eps * scale_const_val);
                        }
                    }

                    return true;
                }
            }
        }

        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(norm_m, "ScalarMulShareTransformation");
    this->register_matcher(m, callback);
}

ov::pass::activations_scaling::MoveDownScalarMul::MoveDownScalarMul() {
    MATCHER_SCOPE(MoveDownScalarMul);

    auto activation_b_m = any_input(is_non_const_node);
    auto mul_const_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(is_scalar_node);
    auto mul_b_m = wrap_type<ov::op::v1::Multiply>({activation_b_m, mul_const_m});
    auto activation_a_m = any_input(is_non_const_node);
    auto mul_a_m = wrap_type<ov::op::v1::Multiply>({activation_a_m, mul_b_m});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto mul_a = pattern_map.at(mul_a_m).get_node_shared_ptr();
        auto mul_b = pattern_map.at(mul_b_m).get_node_shared_ptr();
        auto output_type = mul_a->get_output_element_type(0);

        auto new_mul_a = std::make_shared<ov::op::TypeRelaxed<ov::op::v1::Multiply>>(
            std::vector<element::Type>{output_type, output_type},
            std::vector<element::Type>{output_type},
            ov::op::TemporaryReplaceOutputType(pattern_map.at(activation_a_m), output_type).get(),
            ov::op::TemporaryReplaceOutputType(pattern_map.at(activation_b_m), output_type).get());
        new_mul_a->set_friendly_name(mul_a->get_friendly_name() + "_mm");
        ov::copy_runtime_info(mul_a, new_mul_a);

        auto new_mul_b = std::make_shared<ov::op::TypeRelaxed<ov::op::v1::Multiply>>(
            std::vector<element::Type>{output_type, output_type},
            std::vector<element::Type>{output_type},
            ov::op::TemporaryReplaceOutputType(new_mul_a->output(0), output_type).get(),
            ov::op::TemporaryReplaceOutputType(pattern_map.at(mul_const_m), output_type).get());
        new_mul_b->set_friendly_name(mul_b->get_friendly_name() + "_mm");
        ov::copy_runtime_info(mul_b, new_mul_b);

        ov::replace_node(mul_a, new_mul_b);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul_a_m, "MoveDownScalarMul");
    this->register_matcher(m, callback);
}
