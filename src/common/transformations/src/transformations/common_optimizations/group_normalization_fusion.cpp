// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/group_normalization_fusion.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/group_normalization.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::GroupNormalizationFusion::GroupNormalizationFusion() {
    MATCHER_SCOPE(GroupNormalizationFusion);

    auto input_m = ov::pass::pattern::any_input();

    auto pre_mvn_shape_const_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto pre_mvn_reshape_m = ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({input_m, pre_mvn_shape_const_m});

    auto axes_const_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto mvn_m = ov::pass::pattern::wrap_type<ov::op::v6::MVN>({pre_mvn_reshape_m, axes_const_m});

    auto instance_norm_gamma_m = ov::pass::pattern::any_input();
    auto instance_norm_gamma_multiply_m =
        ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({mvn_m, instance_norm_gamma_m});
    auto instance_norm_opt_gamma_m =
        std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{mvn_m, instance_norm_gamma_multiply_m});

    auto instance_norm_beta_m = ov::pass::pattern::any_input();
    auto instance_norm_beta_add_m =
        ov::pass::pattern::wrap_type<ov::op::v1::Add>({instance_norm_opt_gamma_m, instance_norm_beta_m});
    auto instance_norm_opt_gamma_opt_beta_m = std::make_shared<ov::pass::pattern::op::Or>(
        ov::OutputVector{instance_norm_opt_gamma_m, instance_norm_beta_add_m});

    auto post_instance_norm_shape_m = ov::pass::pattern::any_input();
    auto post_instance_norm_reshape_m = ov::pass::pattern::wrap_type<ov::op::v1::Reshape>(
        {instance_norm_opt_gamma_opt_beta_m, post_instance_norm_shape_m});

    auto group_norm_gamma_m = ov::pass::pattern::any_input();
    auto group_norm_gamma_multiply_m =
        ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({post_instance_norm_reshape_m, group_norm_gamma_m});

    auto group_norm_beta_m = ov::pass::pattern::any_input();
    auto group_norm_beta_add_m =
        ov::pass::pattern::wrap_type<ov::op::v1::Add>({group_norm_gamma_multiply_m, group_norm_beta_m});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto input = pattern_map.at(input_m);
        auto input_ps = input.get_partial_shape();

        auto T = input.get_element_type();

        // this pattern supports only real and not quantized data types
        if ((!T.is_real()) || (T.is_quantized()))
            return false;

        // expecting at least 2D tensor as pattern input:
        // (batch_size, num_channels, ...)
        if (input_ps.size() < 2)
            return false;
        // channel dimension has to be static, all other dimensions in input can be dynamic
        if (input_ps[1].is_dynamic())
            return false;

        auto pre_mvn_reshape_out = pattern_map.at(pre_mvn_reshape_m);
        auto pre_mvn_reshape_out_ps = pre_mvn_reshape_out.get_partial_shape();

        // expecting 3D static tensor as pre-MVN reshape input:
        // (batch_size, num_groups, -1)
        if (pre_mvn_reshape_out_ps.size() != 3)
            return false;

        // channel dimension has to be static, all other dimensions in pre-MVN reshape can be dynamic
        if (pre_mvn_reshape_out_ps[1].is_dynamic())
            return false;

        auto num_channels = input_ps[1].get_max_length();
        auto num_groups = pre_mvn_reshape_out_ps[1].get_max_length();

        // number of channels has to be divisible by number of groups
        if (num_channels % num_groups != 0)
            return false;
        auto channels_to_groups_ratio = num_channels / num_groups;

        // MVN input has to have at least two dimensions:
        // (batch_size, num_groups, ...)
        if (pre_mvn_reshape_out_ps.size() < 2)
            return false;

        // first dimension of MVN input (batch_size) has to be the same
        // as in pattern input
        if (input_ps[0].get_max_length() != pre_mvn_reshape_out_ps[0].get_max_length())
            return false;

        auto post_instance_norm_reshape_out = pattern_map.at(post_instance_norm_reshape_m);
        auto post_instance_norm_reshape_out_ps = post_instance_norm_reshape_out.get_partial_shape();

        // post instance norm shape has to be same as in pattern input:
        // (batch_size, num_channels, height, width)
        if (post_instance_norm_reshape_out_ps != input_ps)
            return false;

        auto group_norm_gamma = pattern_map.at(group_norm_gamma_m);
        auto group_norm_gamma_ps = group_norm_gamma.get_partial_shape();

        // group_norm_gamma has to share the same data type as
        // pattern input
        if (group_norm_gamma.get_element_type() != T)
            return false;

        // group_norm_gamma has to be static
        if (group_norm_gamma_ps.is_dynamic())
            return false;

        // number of elements in group_norm_gamma must be equal to
        // number of channels
        if (ov::shape_size(group_norm_gamma.get_shape()) != num_channels)
            return false;

        auto group_norm_beta = pattern_map.at(group_norm_beta_m);
        auto group_norm_beta_ps = group_norm_beta.get_partial_shape();

        // group_norm_beta has to share the same data type as
        // pattern input
        if (group_norm_beta.get_element_type() != T)
            return false;

        // group_norm_beta has to be static
        if (group_norm_beta_ps.is_dynamic())
            return false;

        // number of elements in group_norm_beta must be equal to
        // number of channels
        if (ov::shape_size(group_norm_beta.get_shape()) != num_channels)
            return false;

        auto expected_param_shape = ov::PartialShape({num_channels});

        std::shared_ptr<ov::Node> group_norm_gamma_1d_m = std::make_shared<ov::op::v0::Squeeze>(group_norm_gamma);
        auto group_norm_gamma_1d_out = group_norm_gamma_1d_m->get_default_output();
        auto group_norm_gamma_1d_out_ps = group_norm_gamma_1d_out.get_partial_shape();

        if (group_norm_gamma_1d_out_ps != expected_param_shape)
            return false;

        std::shared_ptr<ov::Node> group_norm_beta_1d_m = std::make_shared<ov::op::v0::Squeeze>(group_norm_beta);
        auto group_norm_beta_1d_out = group_norm_beta_1d_m->get_default_output();
        auto group_norm_beta_1d_out_ps = group_norm_beta_1d_out.get_partial_shape();

        if (group_norm_beta_1d_out_ps != expected_param_shape)
            return false;

        std::shared_ptr<ov::Node> instance_norm_beta_1d_m = nullptr;
        if (pattern_map.count(instance_norm_beta_m) > 0) {
            auto instance_norm_beta = pattern_map.at(instance_norm_beta_m);
            auto instance_norm_beta_ps = group_norm_beta.get_partial_shape();

            // instance_norm_beta has to share the same data type as
            // pattern input
            if (instance_norm_beta.get_element_type() != T)
                return false;

            // instance_norm_beta has to be static
            if (instance_norm_beta_ps.is_dynamic())
                return false;

            // number of elements in instance_norm_beta must be equal to
            // number of groups
            if (ov::shape_size(instance_norm_beta.get_shape()) != num_groups)
                return false;

            // ensure that instance_norm_beta will have shape compatible
            // with group_norm parameters, i.e. 1D vector of shape (num_channels)
            if (ov::shape_size(instance_norm_beta.get_shape()) == 1) {
                auto shape_1d_const_m = op::v0::Constant::create(element::i64, Shape{1}, {1});
                instance_norm_beta_1d_m =
                    std::make_shared<ov::op::v1::Reshape>(instance_norm_beta, shape_1d_const_m, true);
            } else {
                instance_norm_beta_1d_m = std::make_shared<ov::op::v0::Squeeze>(instance_norm_beta);
            }
            ov::OutputVector instance_norm_beta_concat_inputs;
            for (auto i = 0; i < channels_to_groups_ratio; i++)
                instance_norm_beta_concat_inputs.push_back(instance_norm_beta_1d_m);
            instance_norm_beta_1d_m = std::make_shared<ov::op::v0::Concat>(instance_norm_beta_concat_inputs, 0);
            auto instance_norm_beta_1d_out = instance_norm_beta_1d_m->get_default_output();
            auto instance_norm_beta_1d_ps = instance_norm_beta_1d_out.get_partial_shape();
            if (instance_norm_beta_1d_ps != expected_param_shape)
                return false;
        }

        if (pattern_map.count(instance_norm_gamma_m) > 0) {
            auto instance_norm_gamma = pattern_map.at(instance_norm_gamma_m);
            auto instance_norm_gamma_ps = group_norm_beta.get_partial_shape();

            // instance_norm_gamma has to share the same data type as
            // pattern input
            if (instance_norm_gamma.get_element_type() != T)
                return false;

            // instance_norm_gamma has to be static
            if (instance_norm_gamma_ps.is_dynamic())
                return false;

            // number of elements in instance_norm_gamma must be equal to
            // number of groups
            if (ov::shape_size(instance_norm_gamma.get_shape()) != num_groups)
                return false;

            // ensure that instance_norm_gamma will have shape compatible
            // with group_norm parameters, i.e. 1D vector of shape (num_channels)
            std::shared_ptr<ov::Node> instance_norm_gamma_1d_m = nullptr;
            if (ov::shape_size(instance_norm_gamma.get_shape()) == 1) {
                auto shape_1d_const_m = op::v0::Constant::create(element::i64, Shape{1}, {1});
                instance_norm_gamma_1d_m =
                    std::make_shared<ov::op::v1::Reshape>(instance_norm_gamma, shape_1d_const_m, true);
            } else {
                instance_norm_gamma_1d_m = std::make_shared<ov::op::v0::Squeeze>(instance_norm_gamma);
            }
            ov::OutputVector instance_norm_gamma_concat_inputs;
            for (auto i = 0; i < channels_to_groups_ratio; i++)
                instance_norm_gamma_concat_inputs.push_back(instance_norm_gamma_1d_m);
            instance_norm_gamma_1d_m = std::make_shared<ov::op::v0::Concat>(instance_norm_gamma_concat_inputs, 0);
            auto instance_norm_gamma_1d_out = instance_norm_gamma_1d_m->get_default_output();
            auto instance_norm_gamma_1d_ps = instance_norm_gamma_1d_out.get_partial_shape();
            if (instance_norm_gamma_1d_ps != expected_param_shape)
                return false;

            // group_norm_gamma /= instance_norm_gamma
            group_norm_gamma_1d_m =
                std::make_shared<ov::op::v1::Divide>(group_norm_gamma_1d_m, instance_norm_gamma_1d_m);

            if (pattern_map.count(instance_norm_beta_m) > 0) {
                // group_norm_beta -= group_norm_gamma * instance_norm_beta / instance_norm_gamma
                auto group_norm_beta_corr_multiply_m =
                    std::make_shared<ov::op::v1::Multiply>(group_norm_gamma_1d_m, instance_norm_beta_1d_m);
                auto group_norm_beta_corr_divide_m =
                    std::make_shared<ov::op::v1::Divide>(group_norm_beta_corr_multiply_m, instance_norm_gamma_1d_m);
                group_norm_beta_1d_m =
                    std::make_shared<ov::op::v1::Subtract>(group_norm_beta_1d_m, group_norm_beta_corr_divide_m);
            }
        } else {
            if (pattern_map.count(instance_norm_beta_m) > 0) {
                // group_norm_beta -= group_norm_gamma * instance_norm_beta
                auto group_norm_beta_corr_multiply_m =
                    std::make_shared<ov::op::v1::Multiply>(group_norm_gamma_1d_m, instance_norm_beta_1d_m);
                group_norm_beta_1d_m =
                    std::make_shared<ov::op::v1::Subtract>(group_norm_beta_1d_m, group_norm_beta_corr_multiply_m);
            }
        }

        // we need to be able to cast mvn to MVN layer type
        // in order to read actual epsilon value
        auto mvn_out = pattern_map.at(mvn_m);
        auto mvn = std::dynamic_pointer_cast<ov::op::v6::MVN>(mvn_out.get_node_shared_ptr());
        auto epsilon = mvn->get_eps();

        // we can finally create GroupNormalization op
        std::shared_ptr<ov::Node> group_norm = std::make_shared<ov::op::v12::GroupNormalization>(input,
                                                                                                 group_norm_gamma_1d_m,
                                                                                                 group_norm_beta_1d_m,
                                                                                                 num_groups,
                                                                                                 epsilon);

        // and do actual graph substitution
        group_norm->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), group_norm);
        ov::replace_node(m.get_match_root(), group_norm);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(group_norm_beta_add_m, matcher_name);
    this->register_matcher(m, callback);
}
