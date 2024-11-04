// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/activations_scaling.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/group_normalization.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
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
    auto node = std::dynamic_pointer_cast<ov::op::v0::Constant>(output.get_node_shared_ptr());
    if (node) {
        return false;
    } else {
        return true;
    }
};
}  // namespace

using namespace ov::pass::activations_scaling;
using namespace ov::pass::pattern;
using ov::pass::pattern::op::Or;

// Add scale_down and scale_up layers around Convolution and MatMul nodes
// Conv/MatMul
//    ==>
// Multiply(scale_down by scale_factor) --> Conv/MatMul --> Multiply(scale_up by scale_factor)
ov::pass::activations_scaling::ScaleDownSingleLayer::ScaleDownSingleLayer(float scale_factor) {
    MATCHER_SCOPE(ScaleDownSingleLayer);

    auto activation_m = any_input();
    auto weights_m = any_input();
    auto convolution_m = wrap_type<ov::op::v1::Convolution>({activation_m, weights_m});
    auto matmul_m = wrap_type<ov::op::v0::MatMul>({activation_m, weights_m});
    auto scaled_op_m = std::make_shared<Or>(OutputVector{convolution_m, matmul_m});

    ov::Shape scale_const_shape = {1};
    std::vector<float> scale_down_value = {1.f / scale_factor};
    std::shared_ptr<ov::Node> scale_down_const_f16 =
        std::make_shared<ov::op::v0::Constant>(ov::element::f16, scale_const_shape, scale_down_value);
    std::shared_ptr<ov::Node> scale_down_const_f32 =
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, scale_const_shape, scale_down_value);
    std::vector<float> scale_up_value = {scale_factor};
    std::shared_ptr<ov::Node> scale_up_const_f16 =
        std::make_shared<ov::op::v0::Constant>(ov::element::f16, scale_const_shape, scale_up_value);
    std::shared_ptr<ov::Node> scale_up_const_f32 =
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, scale_const_shape, scale_up_value);

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        OPENVINO_ASSERT(pattern_map.count(convolution_m) || pattern_map.count(matmul_m));

        std::shared_ptr<ov::op::Op> scaled_op = nullptr;

        if (pattern_map.count(convolution_m))
            scaled_op = std::dynamic_pointer_cast<ov::op::Op>(pattern_map.at(convolution_m).get_node_shared_ptr());

        if (pattern_map.count(matmul_m))
            scaled_op = std::dynamic_pointer_cast<ov::op::Op>(pattern_map.at(matmul_m).get_node_shared_ptr());

        if (transformation_callback(scaled_op))
            return false;

        auto scale_down = std::make_shared<ov::op::v1::Multiply>(
            scaled_op->input(0).get_source_output(),
            (scaled_op->input(0).get_element_type() == ov::element::f32) ? scale_down_const_f32 : scale_down_const_f16);
        scale_down->set_friendly_name(scaled_op->get_friendly_name() + "_scale_down");
        ov::copy_runtime_info(scaled_op, scale_down);
        scaled_op->input(0).replace_source_output(scale_down->output(0));

        auto child = scaled_op->get_output_target_inputs(0).begin()->get_node();
        if (scaled_op->get_output_target_inputs(0).size() == 1 && ov::is_type<ov::op::v1::Add>(child)) {
            auto add = child->shared_from_this();
            auto target_inputs = add->get_output_target_inputs(0);
            auto scale_down_bias = std::make_shared<ov::op::v1::Multiply>(
                add->input(1).get_source_output(),
                (add->input(1).get_element_type() == ov::element::f32) ? scale_down_const_f32 : scale_down_const_f16);
            scale_down_bias->set_friendly_name(add->get_friendly_name() + "_scale_down");
            ov::copy_runtime_info(add, scale_down_bias);
            add->input(1).replace_source_output(scale_down_bias->output(0));

            auto scale_up = register_new_node<ov::op::v1::Multiply>(
                add->output(0),
                (add->output(0).get_element_type() == ov::element::f32) ? scale_up_const_f32 : scale_up_const_f16);
            scale_up->set_friendly_name(scaled_op->get_friendly_name() + "_scale_up");
            ov::copy_runtime_info(scaled_op, scale_up);
            for (auto& in : target_inputs) {
                in.replace_source_output(scale_up);
            }
        } else {
            auto target_inputs = scaled_op->get_output_target_inputs(0);
            auto scale_up = register_new_node<ov::op::v1::Multiply>(
                scaled_op->output(0),
                (scaled_op->output(0).get_element_type() == ov::element::f32) ? scale_up_const_f32
                                                                              : scale_up_const_f16);
            scale_up->set_friendly_name(scaled_op->get_friendly_name() + "_scale_up");
            ov::copy_runtime_info(scaled_op, scale_up);
            for (auto& in : target_inputs) {
                in.replace_source_output(scale_up);
            }
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(scaled_op_m, "ScaleDownSingleLayer");
    this->register_matcher(m, callback);
}

// MulMulAddTransformation makes the target pattern to be easy to be merged with followig nodes.
//
// input_a   const_a   input_b  const_b          input_a   (const_a/const_b)
//      \      /          \      /                   \      /
//     Multiply_a        Multiply_b          ==>     Multiply_a_mma   input_b
//             \         /                                      \     /
//              \       /                                         Add    const_b
//               \     /                                           |      /
//                 Add                                           Multiply_b_mma
//
// (input_a * const_a) + (input_b * const_b) ==> ((input_a * (const_a / const_b)) + input_b) * const_b
ov::pass::activations_scaling::MulMulAddTransformation::MulMulAddTransformation() {
    MATCHER_SCOPE(MulMulAddTransformation);

    auto activation0_m = any_input(is_non_const_node);
    auto scale_const0_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(is_scalar_node);
    auto mul0_m = wrap_type<ov::op::v1::Multiply>({activation0_m, scale_const0_m});

    auto activation1_m = any_input(is_non_const_node);
    auto scale_const1_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(is_scalar_node);
    auto mul1_m = wrap_type<ov::op::v1::Multiply>({activation1_m, scale_const1_m});

    auto add_m = wrap_type<ov::op::v1::Add>({mul0_m, mul1_m});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        OPENVINO_ASSERT(pattern_map.count(mul0_m));
        OPENVINO_ASSERT(pattern_map.count(mul1_m));
        OPENVINO_ASSERT(pattern_map.count(add_m));

        auto add = std::dynamic_pointer_cast<ov::op::v1::Add>(pattern_map.at(add_m).get_node_shared_ptr());

        if (transformation_callback(add)) {
            return false;
        }
        auto target_inputs = add->get_output_target_inputs(0);

        auto mul0 = add->get_input_source_output(0).get_node_shared_ptr();
        auto mul1 = add->get_input_source_output(1).get_node_shared_ptr();

        size_t const0_index = ov::is_type<ov::op::v0::Constant>(mul0->get_input_source_output(1).get_node()) ? 1 : 0;
        size_t const1_index = ov::is_type<ov::op::v0::Constant>(mul1->get_input_source_output(1).get_node()) ? 1 : 0;

        auto scale_const0 = mul0->get_input_source_output(const0_index).get_node_shared_ptr();
        auto scale_const1 = mul1->get_input_source_output(const1_index).get_node_shared_ptr();

        auto new_mul0 = register_new_node<ov::op::v1::Multiply>(
            mul0->get_input_source_output((const0_index == 0) ? 1 : 0),
            ov::op::util::eltwise_fold<ov::op::v1::Divide>(scale_const0, scale_const1));
        new_mul0->set_friendly_name(mul0->get_friendly_name() + "_mma");
        ov::copy_runtime_info(mul0, new_mul0);

        add->input(0).replace_source_output(new_mul0);
        add->input(1).replace_source_output(mul1->get_input_source_output((const1_index == 0) ? 1 : 0));

        auto new_mul1 = register_new_node<ov::op::v1::Multiply>(add, scale_const1);
        new_mul1->set_friendly_name(mul1->get_friendly_name() + "_mma");
        ov::copy_runtime_info(mul1, new_mul1);

        for (auto& in : target_inputs) {
            in.replace_source_output(new_mul1);
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(add_m, "MulMulAddTransformation");
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
        auto norm = std::dynamic_pointer_cast<ov::op::Op>(pattern_map.at(norm_m).get_node_shared_ptr());

        if (transformation_callback(norm)) {
            return false;
        }

        if (mul && norm) {
            size_t activation_index =
                ov::is_type<ov::op::v0::Constant>(mul->get_input_source_output(1).get_node()) ? 0 : 1;
            norm->input(0).replace_source_output(mul->get_input_source_output(activation_index));
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
        auto norm = std::dynamic_pointer_cast<ov::op::Op>(pattern_map.at(norm_m).get_node_shared_ptr());

        if (transformation_callback(norm)) {
            return false;
        }

        if (mul && norm) {
            size_t activation_index =
                ov::is_type<ov::op::v0::Constant>(mul->get_input_source_output(1).get_node()) ? 0 : 1;
            norm->input(0).replace_source_output(mul->get_input_source_output(activation_index));
            return true;
        }
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(norm_m, "MulMVNTransformation");
    this->register_matcher(m, callback);
}

//        input     const                            input
//           \      /                                  |
//           Multiply              ==>           VariadicSplit
//              |                       const   /   |  const  \      const
//         VariadicSplit                  |    /    |   /      \      /
//         /    |      \               Multiply_a  Multiply_b   Multiply_c
// output_a  output_b   output_c           |           |            |
//                                      output_a    output_b     output_c
ov::pass::activations_scaling::SplitTransformation::SplitTransformation() {
    MATCHER_SCOPE(SplitTransformation);

    auto activation_m = any_input(is_non_const_node);
    auto scale_const_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(is_scalar_node);
    auto mul_m = wrap_type<ov::op::v1::Multiply>({activation_m, scale_const_m});
    auto axis_m = any_input();
    auto split_length_m = any_input();
    auto split_m = wrap_type<ov::op::v1::VariadicSplit>({mul_m, axis_m, split_length_m});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        OPENVINO_ASSERT(pattern_map.count(mul_m));
        OPENVINO_ASSERT(pattern_map.count(split_m));

        auto mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(pattern_map.at(mul_m).get_node_shared_ptr());
        auto split =
            std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(pattern_map.at(split_m).get_node_shared_ptr());

        if (transformation_callback(split)) {
            return false;
        }

        if (mul && split) {
            size_t num_split_outputs = split->get_output_size();

            std::vector<std::set<ov::Input<ov::Node>>> target_inputs;
            target_inputs.resize(num_split_outputs);
            for (size_t i = 0; i < num_split_outputs; i++) {
                target_inputs[i] = split->get_output_target_inputs(i);
            }

            size_t activation_index =
                ov::is_type<ov::op::v0::Constant>(mul->get_input_source_output(1).get_node()) ? 0 : 1;
            size_t const_index = (activation_index == 1) ? 0 : 1;
            split->input(0).replace_source_output(mul->input(activation_index).get_source_output());

            for (size_t i = 0; i < num_split_outputs; i++) {
                auto new_mul = register_new_node<ov::op::v1::Multiply>(split->output(i),
                                                                       mul->input(const_index).get_source_output());
                new_mul->set_friendly_name(mul->get_friendly_name() + "_" + std::to_string(i));
                ov::copy_runtime_info(mul, new_mul);

                for (auto& in : target_inputs[i]) {
                    in.replace_source_output(new_mul);
                }
            }

            return true;
        }
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(split_m, "SplitTransformation");
    this->register_matcher(m, callback);
}

// input    const           input
//    \      /                |
//    Multiply       ==>   Reshape   const
//       |                    |       /
//    Reshape                 Multiply
ov::pass::activations_scaling::ReshapeTransformation::ReshapeTransformation() {
    MATCHER_SCOPE(ReshapeTransformation);

    auto activation_m = any_input(is_non_const_node);
    auto scale_const_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(is_scalar_node);
    auto mul_m = wrap_type<ov::op::v1::Multiply>({activation_m, scale_const_m});
    auto axes_m = any_input();
    auto reshape_m = wrap_type<ov::op::v1::Reshape>({mul_m, axes_m});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        OPENVINO_ASSERT(pattern_map.count(mul_m));
        OPENVINO_ASSERT(pattern_map.count(reshape_m));

        auto scale_const =
            std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(scale_const_m).get_node_shared_ptr());
        auto mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(pattern_map.at(mul_m).get_node_shared_ptr());
        auto reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(pattern_map.at(reshape_m).get_node_shared_ptr());

        if (transformation_callback(reshape)) {
            return false;
        }

        if (scale_const && mul && reshape) {
            auto target_inputs = reshape->get_output_target_inputs(0);
            size_t activation_index =
                ov::is_type<ov::op::v0::Constant>(mul->get_input_source_output(1).get_node()) ? 0 : 1;
            reshape->input(0).replace_source_output(mul->input(activation_index).get_source_output());

            auto new_mul = register_new_node<ov::op::v1::Multiply>(reshape, scale_const);
            new_mul->set_friendly_name(mul->get_friendly_name() + "_r");
            ov::copy_runtime_info(mul, new_mul);

            for (auto& in : target_inputs) {
                in.replace_source_output(new_mul);
            }

            return true;
        }
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(reshape_m, "ReshapeTransformation");
    this->register_matcher(m, callback);
}

// MulMulAddTransformation makes the target pattern to be easy to be merged with other nodes.
//
// input_a   const_a   input_b  const_b          input_a     input_b
//      \      /          \      /                   \        /
//     Multiply_a        Multiply_b          ==>     Multiply_c       (const_a * const_b)
//             \         /                                    \         /
//              \       /                                   Multiply_c_mmm
//               \     /
//              Multiply_c
//
// (input_a * const_a) * (input_b * const_b) ==> (input_a * input_b) * (const_a * const_b)
ov::pass::activations_scaling::MulMulMulTransformation::MulMulMulTransformation() {
    MATCHER_SCOPE(MulMulMulTransformation);

    auto activation0_m = any_input(is_non_const_node);
    auto scale_const0_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(is_scalar_node);
    auto mul0_m = wrap_type<ov::op::v1::Multiply>({activation0_m, scale_const0_m});

    auto activation1_m = any_input(is_non_const_node);
    auto scale_const1_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(is_scalar_node);
    auto mul1_m = wrap_type<ov::op::v1::Multiply>({activation1_m, scale_const1_m});

    auto mul2_m = wrap_type<ov::op::v1::Multiply>({mul0_m, mul1_m});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        OPENVINO_ASSERT(pattern_map.count(mul0_m));
        OPENVINO_ASSERT(pattern_map.count(mul1_m));
        OPENVINO_ASSERT(pattern_map.count(mul2_m));

        auto mul2 = std::dynamic_pointer_cast<ov::op::v1::Multiply>(pattern_map.at(mul2_m).get_node_shared_ptr());

        if (transformation_callback(mul2)) {
            return false;
        }
        auto target_inputs = mul2->get_output_target_inputs(0);

        auto mul0 = mul2->get_input_source_output(0).get_node_shared_ptr();
        auto mul1 = mul2->get_input_source_output(1).get_node_shared_ptr();

        size_t const0_index = ov::is_type<ov::op::v0::Constant>(mul0->get_input_source_output(1).get_node()) ? 1 : 0;
        size_t const1_index = ov::is_type<ov::op::v0::Constant>(mul1->get_input_source_output(1).get_node()) ? 1 : 0;

        auto scale_const0 = mul0->get_input_source_output(const0_index).get_node_shared_ptr();
        auto scale_const1 = mul1->get_input_source_output(const1_index).get_node_shared_ptr();

        mul2->input(0).replace_source_output(mul0->get_input_source_output((const0_index == 0) ? 1 : 0));
        mul2->input(1).replace_source_output(mul1->get_input_source_output((const1_index == 0) ? 1 : 0));

        auto new_mul = register_new_node<ov::op::v1::Multiply>(
            mul2,
            ov::op::util::eltwise_fold<ov::op::v1::Multiply>(scale_const0, scale_const1));
        new_mul->set_friendly_name(mul2->get_friendly_name() + "_mmm");
        ov::copy_runtime_info(mul2, new_mul);

        for (auto& in : target_inputs) {
            in.replace_source_output(new_mul);
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul2_m, "MulMulMulTransformation");
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
ov::pass::activations_scaling::ConcatTransformation::ConcatTransformation() {
    MATCHER_SCOPE(ConcatTransformation);

    auto concat_m = wrap_type<ov::op::v0::Concat>();

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        OPENVINO_ASSERT(pattern_map.count(concat_m));

        auto concat = std::dynamic_pointer_cast<ov::op::v0::Concat>(pattern_map.at(concat_m).get_node_shared_ptr());

        if (transformation_callback(concat_m)) {
            return false;
        }

        // check if all inputs are Multiply with scalar operand
        ov::Output<ov::Node> last_dep_const;
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
            if (!is_scalar_node(last_dep_const)) {
                return false;
            }
        }

        auto target_inputs = concat->get_output_target_inputs(0);

        for (auto& input : concat->inputs()) {
            auto dep_node = input.get_source_output().get_node_shared_ptr();
            auto dep_input0 = dep_node->input(0).get_source_output().get_node();
            size_t const_index = ov::is_type<ov::op::v0::Constant>(dep_input0) ? 0 : 1;
            size_t activation_index = ov::is_type<ov::op::v0::Constant>(dep_input0) ? 1 : 0;

            auto new_mul = register_new_node<ov::op::v1::Multiply>(
                dep_node->input(activation_index).get_source_output(),
                ov::op::util::eltwise_fold<ov::op::v1::Divide>(dep_node->input(const_index).get_source_output(),
                                                               last_dep_const));
            new_mul->set_friendly_name(dep_node->get_friendly_name() + "_c");
            ov::copy_runtime_info(dep_node, new_mul);

            input.replace_source_output(new_mul);
        }

        auto new_mul = register_new_node<ov::op::v1::Multiply>(concat, last_dep_const);
        new_mul->set_friendly_name(concat->get_friendly_name() + "_c");
        ov::copy_runtime_info(concat, new_mul);

        for (auto& in : target_inputs) {
            in.replace_source_output(new_mul);
        }

        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(concat_m, "ConcatTransformation");
    this->register_matcher(m, callback);
}

bool ov::pass::ActivationsScaling::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(ActivationsScaling);

    if (m_scale_factor <= 0.f)
        return false;

    ov::pass::Manager manager(get_pass_config(), "ActivationsScaling");
    manager.set_per_pass_validation(false);

    manager.register_pass<ScaleDownSingleLayer>(m_scale_factor);
    manager.register_pass<LinOpSequenceFusion>();
    manager.register_pass<MulGroupNormTransformation>();
    manager.register_pass<LinOpSequenceFusion>();
    manager.register_pass<MulMulAddTransformation>();
    manager.register_pass<MulGroupNormTransformation>();
    manager.register_pass<SplitTransformation>();
    manager.register_pass<ReshapeTransformation>();
    manager.register_pass<MulMulMulTransformation>();
    manager.register_pass<MulMulAddTransformation>();
    manager.register_pass<ConcatTransformation>();
    manager.register_pass<MulMVNTransformation>();
    manager.register_pass<MulMulAddTransformation>();
    manager.register_pass<MulMVNTransformation>();

    manager.run_passes(f);

    return true;
}
