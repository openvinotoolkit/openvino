// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/activations_scaling.hpp"

#include <memory>

#include "itt.hpp"
#include "low_precision/network_helper.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
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
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/common_optimizations/lin_op_sequence_fusion.hpp"
#include "transformations/utils/utils.hpp"

namespace {
const auto is_scalar_node = [](const ov::Output<ov::Node>& output) -> bool {
    const auto shape = output.get_partial_shape();
    if (shape.is_dynamic() || shape.rank().is_dynamic())
        return false;
    if (std::all_of(shape.begin(), shape.end(), [](const ov::Dimension& dimension) {
            return dimension == 1ul;
        }))
        return true;
    return false;
};

const auto is_non_const_node = [](const ov::Output<ov::Node>& output) -> bool {
    return !ov::is_type<ov::op::v0::Constant>(output.get_node());
};
}  // namespace

void ov::pass::activations_scaling::mark_as_scale_down_node(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info[ScaleDownNode::get_type_info_static()] = ScaleDownNode();
}

bool ov::pass::activations_scaling::is_scale_down_node(const std::shared_ptr<const Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.find(ScaleDownNode::get_type_info_static()) != rt_info.end();
}

using namespace ov::pass::activations_scaling;
using namespace ov::pass::pattern;
using ov::pass::pattern::op::Or;

// Add scale_down and scale_up layers around Convolution and MatMul nodes
// Conv/MatMul
//    ==>
// Multiply(scale_down by scale_factor) --> Conv/MatMul --> Multiply(scale_up by scale_factor)
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

        OPENVINO_ASSERT(pattern_map.count(convolution_m) || pattern_map.count(matmul_m));

        // scale_down and scale_up layers will be added around scaled_op
        std::shared_ptr<ov::op::Op> scaled_op = nullptr;

        if (pattern_map.count(convolution_m))
            scaled_op = std::dynamic_pointer_cast<ov::op::Op>(pattern_map.at(convolution_m).get_node_shared_ptr());

        if (pattern_map.count(matmul_m))
            scaled_op = std::dynamic_pointer_cast<ov::op::Op>(pattern_map.at(matmul_m).get_node_shared_ptr());

        if (transformation_callback(scaled_op))
            return false;

        // in the case of decompressed_to_f32 nodes, scale_up layer will be added after Convert node.
        bool keep_precision = false;
        std::shared_ptr<ov::op::Op> output_of_scaled_op = scaled_op;
        auto child_node = scaled_op->get_output_target_inputs(0).begin()->get_node();
        if (scaled_op->get_output_target_inputs(0).size() == 1 && ov::is_type<ov::op::v0::Convert>(child_node) &&
            ov::fp16_compression_is_disabled(child_node->shared_from_this()) &&
            ov::pass::constant_folding_is_disabled(child_node->shared_from_this())) {
            output_of_scaled_op = std::dynamic_pointer_cast<ov::op::Op>(child_node->shared_from_this());
            child_node = output_of_scaled_op->get_output_target_inputs(0).begin()->get_node();
            keep_precision = true;
        }

        const ov::Shape scale_shape = {};
        const std::vector<float> scale_down_value = {1.f / scale_factor};
        const std::vector<float> scale_up_value = {scale_factor};
        auto output_prec = output_of_scaled_op->output(0).get_element_type();

        // adding a scale_down layer before the target node
        auto scale_down = std::make_shared<ov::op::v1::Multiply>(
            scaled_op->input(0).get_source_output(),
            std::make_shared<ov::op::v0::Constant>(scaled_op->input(0).get_element_type(),
                                                   scale_shape,
                                                   scale_down_value));
        scale_down->set_friendly_name(scaled_op->get_friendly_name() + "_scale_down");
        ov::copy_runtime_info(scaled_op, scale_down);
        mark_as_scale_down_node(scale_down);

        if (scale_down->output(0).get_element_type() != scaled_prec && !keep_precision) {
            auto convert_prec0 = std::make_shared<ov::op::v0::Convert>(scale_down->output(0), scaled_prec);
            scaled_op->input(0).replace_source_output(convert_prec0->output(0));
        } else {
            scaled_op->input(0).replace_source_output(scale_down->output(0));
        }
        if (scaled_op->input(1).get_element_type() != scaled_prec && !keep_precision) {
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
            auto scale_down_bias = std::make_shared<ov::op::v1::Multiply>(
                add->input(bias_index).get_source_output(),
                std::make_shared<ov::op::v0::Constant>(add->input(bias_index).get_element_type(),
                                                       scale_shape,
                                                       scale_down_value));
            scale_down_bias->set_friendly_name(add->get_friendly_name() + "_scale_down");
            ov::copy_runtime_info(add, scale_down_bias);
            if (scale_down_bias->output(0).get_element_type() != scaled_prec && !keep_precision) {
                auto convert_bias_prec = std::make_shared<ov::op::v0::Convert>(scale_down_bias->output(0), scaled_prec);
                add->input(bias_index).replace_source_output(convert_bias_prec->output(0));
            } else {
                add->input(bias_index).replace_source_output(scale_down_bias->output(0));
            }
            add->revalidate_and_infer_types();
            if (add->output(0).get_element_type() != output_prec && !keep_precision) {
                output_of_scaled_op = std::make_shared<ov::op::v0::Convert>(add->output(0), output_prec);
            } else {
                output_of_scaled_op = std::dynamic_pointer_cast<ov::op::Op>(add);
            }
        } else {
            target_inputs = output_of_scaled_op->get_output_target_inputs(0);
            if (output_of_scaled_op->output(0).get_element_type() != output_prec && !keep_precision) {
                output_of_scaled_op =
                    std::make_shared<ov::op::v0::Convert>(output_of_scaled_op->output(0), output_prec);
            }
        }

        auto scale_up = register_new_node<ov::op::v1::Multiply>(
            output_of_scaled_op->output(0),
            std::make_shared<ov::op::v0::Constant>(output_of_scaled_op->output(0).get_element_type(),
                                                   scale_shape,
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

// ScaleDownFusion merges multiple scale_down layers into one.
//
//               input                      input
//               /   \           ==>          |
//           Mul_a   Mul_b                  Mul_a
//             |       |                    /   |
//           op_a    op_b                op_a  op_b
ov::pass::activations_scaling::ScaleDownFusion::ScaleDownFusion() {
    MATCHER_SCOPE(ScaleDownFusion);

    const auto is_scale_down_mul = [](const ov::Output<ov::Node>& output) -> bool {
        return is_scale_down_node(output.get_node_shared_ptr());
    };

    auto activation_m = any_input();
    auto scale_const_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(is_scalar_node);
    auto mul_m = wrap_type<ov::op::v1::Multiply>({activation_m, scale_const_m}, is_scale_down_mul);

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(pattern_map.at(mul_m).get_node_shared_ptr());
        auto parent = mul->get_input_node_shared_ptr(0);
        if (parent->get_output_size() > 1)
            return false;

        auto children = parent->get_users();
        size_t num_scaled_down_nodes = 0;
        for (const auto& child : children) {
            if (is_scale_down_node(child))
                num_scaled_down_nodes += 1;
        }

        if (num_scaled_down_nodes < 2)
            return false;

        if (transformation_callback(mul))
            return false;

        for (const auto& child : children) {
            if (is_scale_down_node(child)) {
                for (auto& target : child->get_output_target_inputs(0)) {
                    target.replace_source_output(mul->output(0));
                }
            }
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul_m, "ScaleDownFusion");
    this->register_matcher(m, callback);
}

// GroupNormalization has the following property.
//
// GroupNorm(input * const_a) = GroupNorm(input)
//
// So, we can skip Multiply that is connected to GroupNormalization.
//
// input --> Multiply --> GroupNormalization
//   ==>
// input --> GroupNormalization
ov::pass::activations_scaling::MulGroupNormTransformation::MulGroupNormTransformation() {
    MATCHER_SCOPE(MulGroupNormTransformation);

    auto activation_m = any_input(is_non_const_node);
    auto scale_const_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(is_scalar_node);
    auto mul_m = wrap_type<ov::op::v1::Multiply>({activation_m, scale_const_m});
    auto norm_scale_m = any_input();
    auto norm_bias_m = any_input();
    auto norm_m = wrap_type<ov::op::v12::GroupNormalization>({mul_m, norm_scale_m, norm_bias_m});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        OPENVINO_ASSERT(pattern_map.count(mul_m));
        OPENVINO_ASSERT(pattern_map.count(norm_m));

        auto mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(pattern_map.at(mul_m).get_node_shared_ptr());
        auto norm =
            std::dynamic_pointer_cast<ov::op::v12::GroupNormalization>(pattern_map.at(norm_m).get_node_shared_ptr());

        if (transformation_callback(norm)) {
            return false;
        }

        if (mul && norm) {
            size_t activation_index =
                ov::is_type<ov::op::v0::Constant>(mul->get_input_source_output(1).get_node()) ? 0 : 1;
            auto activation = mul->get_input_source_output(activation_index);
            if (ov::is_type<ov::op::v0::Convert>(activation.get_node()))
                activation = activation.get_node()->get_input_source_output(0);
            auto newGroupNorm = std::make_shared<ov::op::TypeRelaxed<ov::op::v12::GroupNormalization>>(
                ov::op::v12::GroupNormalization(activation,
                                                norm->get_input_source_output(1),
                                                norm->get_input_source_output(2),
                                                norm->get_num_groups(),
                                                norm->get_epsilon()),
                norm->get_output_element_type(0));
            ov::copy_runtime_info(norm, newGroupNorm);
            ov::replace_node(norm, newGroupNorm);
            return true;
        }
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(norm_m, "MulGroupNormTransformation");
    this->register_matcher(m, callback);
}

// MVN has the following property.
//
// MVN(input * const_a) = MVN(input)
//
// So, we can skip Multiply that is connected to MVN.
//
// input --> Multiply --> MVN
//   ==>
// input --> MVN
ov::pass::activations_scaling::MulMVNTransformation::MulMVNTransformation() {
    MATCHER_SCOPE(MulMVNTransformation);

    auto activation_m = any_input(is_non_const_node);
    auto scale_const_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(is_scalar_node);
    auto mul_m = wrap_type<ov::op::v1::Multiply>({activation_m, scale_const_m});
    auto norm_axes_m = any_input();
    auto norm_m = wrap_type<ov::op::v6::MVN>({mul_m, norm_axes_m});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        OPENVINO_ASSERT(pattern_map.count(mul_m));
        OPENVINO_ASSERT(pattern_map.count(norm_m));

        auto mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(pattern_map.at(mul_m).get_node_shared_ptr());
        auto norm = std::dynamic_pointer_cast<ov::op::v6::MVN>(pattern_map.at(norm_m).get_node_shared_ptr());

        if (transformation_callback(norm)) {
            return false;
        }

        if (mul && norm) {
            size_t activation_index =
                ov::is_type<ov::op::v0::Constant>(mul->get_input_source_output(1).get_node()) ? 0 : 1;
            auto activation = mul->get_input_source_output(activation_index);
            if (ov::is_type<ov::op::v0::Convert>(activation.get_node()))
                activation = activation.get_node()->get_input_source_output(0);
            auto newMVN =
                std::make_shared<ov::op::TypeRelaxed<ov::op::v6::MVN>>(ov::op::v6::MVN(activation,
                                                                                       norm->get_input_source_output(1),
                                                                                       norm->get_normalize_variance(),
                                                                                       norm->get_eps(),
                                                                                       norm->get_eps_mode()),
                                                                       norm->get_output_element_type(0));
            ov::copy_runtime_info(norm, newMVN);
            ov::replace_node(norm, newMVN);
            return true;
        }
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(norm_m, "MulMVNTransformation");
    this->register_matcher(m, callback);
}

// input_a   const_a   input_b   const_b   input_c   const_c
//    \        /          \        /          \        /
//    Multiply_a          Multiply_b          Multiply_c
//             \              |               /
//              \             |              /
//               ---------- Concat ------------
// ==>
//          (const_a            (const_b             (const_c
// input_a  /const_c)  input_b  /const_c)  input_c   /const_c)
//    \        /          \        /          \        /
//    Multiply_a          Multiply_b          Multiply_c
//             \              |               /
//              \             |              /
//               ---------- Concat ------------
//                            |   const_c
//                            |    /
//                           Multiply
ov::pass::activations_scaling::MulConcatTransformation::MulConcatTransformation() {
    MATCHER_SCOPE(MulConcatTransformation);

    auto concat_m = wrap_type<ov::op::v0::Concat>();

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        OPENVINO_ASSERT(pattern_map.count(concat_m));

        auto concat = std::dynamic_pointer_cast<ov::op::v0::Concat>(pattern_map.at(concat_m).get_node_shared_ptr());

        if (transformation_callback(concat_m)) {
            return false;
        }

        // check if all inputs are Multiply with scalar operand
        ov::Output<ov::Node> last_dep_const = {};
        ov::element::Type last_dep_const_type = ov::element::undefined;
        for (auto& input : concat->inputs()) {
            auto dep_node =
                std::dynamic_pointer_cast<ov::op::v1::Multiply>(input.get_source_output().get_node_shared_ptr());
            if (!dep_node) {
                return false;
            }
            auto dep_const0 = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                dep_node->input(0).get_source_output().get_node_shared_ptr());
            auto dep_const1 = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                dep_node->input(1).get_source_output().get_node_shared_ptr());
            if (!dep_const0 && !dep_const1) {
                return false;
            }
            last_dep_const =
                dep_const0 ? dep_node->input(0).get_source_output() : dep_node->input(1).get_source_output();
            if (!is_scalar_node(last_dep_const))
                return false;
            if (last_dep_const_type != ov::element::undefined &&
                last_dep_const_type != last_dep_const.get_element_type())
                return false;
            last_dep_const_type = last_dep_const.get_element_type();
        }

        auto target_inputs = concat->get_output_target_inputs(0);

        for (auto& input : concat->inputs()) {
            auto dep_node = input.get_source_output().get_node_shared_ptr();
            auto dep_input0 = dep_node->input(0).get_source_output().get_node();
            size_t const_index = ov::is_type<ov::op::v0::Constant>(dep_input0) ? 0 : 1;
            size_t activation_index = ov::is_type<ov::op::v0::Constant>(dep_input0) ? 1 : 0;

            auto dep_type = dep_node->get_output_element_type(0);
            auto new_mul = std::make_shared<ov::op::TypeRelaxed<ov::op::v1::Multiply>>(
                std::vector<element::Type>{dep_type, dep_type},
                std::vector<element::Type>{dep_type},
                ov::op::TemporaryReplaceOutputType(dep_node->input(activation_index).get_source_output(), dep_type)
                    .get(),
                ov::op::TemporaryReplaceOutputType(
                    ov::op::util::eltwise_fold<ov::op::v1::Divide>(dep_node->input(const_index).get_source_output(),
                                                                   last_dep_const),
                    dep_type)
                    .get());
            new_mul->set_friendly_name(dep_node->get_friendly_name() + "_c");
            ov::copy_runtime_info(dep_node, new_mul);

            input.replace_source_output(new_mul);
        }

        auto concat_type = concat->get_output_element_type(0);
        auto new_mul = std::make_shared<ov::op::TypeRelaxed<ov::op::v1::Multiply>>(
            std::vector<element::Type>{concat_type, concat_type},
            std::vector<element::Type>{concat_type},
            ov::op::TemporaryReplaceOutputType(concat->output(0), concat_type).get(),
            ov::op::TemporaryReplaceOutputType(last_dep_const, concat_type).get());
        new_mul->set_friendly_name(concat->get_friendly_name() + "_c");
        ov::copy_runtime_info(concat, new_mul);

        for (auto& in : target_inputs) {
            in.replace_source_output(new_mul);
        }

        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(concat_m, "MulConcatTransformation");
    this->register_matcher(m, callback);
}
