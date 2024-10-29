// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/activations_scaling.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/group_normalization.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
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
}

using namespace ov::pass::pattern;
using ov::pass::pattern::op::Or;

// Add scale_down and scale_up layers around Convolution and MatMul nodes
// Conv/MatMul   ==>   Multiply(scale_down) --> Conv/MatMul --> Multiply(scale_up)
ov::pass::ScaleDownSingleLayer::ScaleDownSingleLayer(float scale_factor) {
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
        scaled_op->input(0).replace_source_output(scale_down->output(0));

        auto child = scaled_op->get_output_target_inputs(0).begin()->get_node();
        if (scaled_op->get_output_target_inputs(0).size() == 1 && ov::is_type<ov::op::v1::Add>(child)) {
            auto add = child->shared_from_this();
            auto scale_down_bias = std::make_shared<ov::op::v1::Multiply>(
                add->input(1).get_source_output(),
                (add->input(1).get_element_type() == ov::element::f32) ? scale_down_const_f32 : scale_down_const_f16);
            add->input(1).replace_source_output(scale_down_bias->output(0));
            
            auto scale_up = std::make_shared<ov::op::v1::Multiply>(
                add->output(0),
                (add->output(0).get_element_type() == ov::element::f32) ? scale_up_const_f32 : scale_up_const_f16);
            ov::replace_node(add, scale_up);
        } else {
            auto scale_up = std::make_shared<ov::op::v1::Multiply>(
                scaled_op->output(0),
                (scaled_op->output(0).get_element_type() == ov::element::f32) ? scale_up_const_f32
                                                                               : scale_up_const_f16);
            ov::replace_node(scaled_op, scale_up);
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(scaled_op_m, "ScaleDownSingleLayer");
    this->register_matcher(m, callback);
}

// MulMulAddFusion makes the target pattern to be easy to be merged with other nodes.
//
// input_a   const_a   input_b  const_b          input_a   (const_a/const_b)
//      \      /          \      /                   \      /
//     Multiply_a        Multiply_b          ==>     Multiply_a    input_b
//             \         /                                  \     /
//              \       /                                     Add     const_b
//               \     /                                       |       /
//                 Add                                        Multiply_c
//
// (input_a * const_a) + (input_b * const_b) ==> ((input_a * (const_a / const_b)) + input_b) * const_b
ov::pass::MulMulAddFusion::MulMulAddFusion() {
    MATCHER_SCOPE(MulMulAddFusion);

    auto activation0_m = any_input();
    auto scale_const0_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(is_scalar_node);
    auto mul0_m = wrap_type<ov::op::v1::Multiply>({activation0_m, scale_const0_m});

    auto activation1_m = any_input();
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

        auto scale_const0 =
            std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(scale_const0_m).get_node_shared_ptr());
        auto mul0 = std::dynamic_pointer_cast<ov::op::v1::Multiply>(pattern_map.at(mul0_m).get_node_shared_ptr());

        auto scale_const1 =
            std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(scale_const1_m).get_node_shared_ptr());
        auto mul1 = std::dynamic_pointer_cast<ov::op::v1::Multiply>(pattern_map.at(mul1_m).get_node_shared_ptr());

        mul0->input(1).replace_source_output(
            ov::op::util::eltwise_fold<ov::op::v1::Divide>(scale_const0, scale_const1));
        add->input(1).replace_source_output(mul1->get_input_source_output(0));

        auto new_mul = register_new_node<ov::op::v1::Multiply>(add, scale_const1);
        replace_node(add, new_mul);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(add_m, "MulMulAddFusion");
    this->register_matcher(m, callback);
}

// GroupNormalization has the following property.
//
// GroupNorm(input * const_a) = GroupNorm(input)
//
// So, we can skip Multiply that is connected to GroupNormalization.
//
// input --> Multiply --> GroupNormalization   ==>   input --> GroupNormalization
ov::pass::MulGroupNormFusion::MulGroupNormFusion() {
    MATCHER_SCOPE(MulGroupNormFusion);

    auto activation_m = any_input();
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
            norm->input(0).replace_source_output(mul->get_input_source_output(0));
            return true;
        }
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(norm_m, "MulGroupNormFusion");
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
    manager.register_pass<MulGroupNormFusion>();
    manager.register_pass<LinOpSequenceFusion>();
    manager.register_pass<MulMulAddFusion>();
    manager.register_pass<MulGroupNormFusion>();

    manager.run_passes(f);

    return true;
}
