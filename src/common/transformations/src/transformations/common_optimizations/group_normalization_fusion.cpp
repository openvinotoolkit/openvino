// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/group_normalization_fusion.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/group_normalization.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::pass::pattern;

ov::pass::GroupNormalizationFusion::GroupNormalizationFusion() {
    MATCHER_SCOPE(GroupNormalizationFusion);

    auto has_real_not_quantized_type = [](const ov::Output<ov::Node>& output) -> bool {
        const auto& T = output.get_element_type();
        return (T.is_real() && (!T.is_quantized()));
    };

    auto has_at_least_2d_shape = [](const ov::Output<ov::Node>& output) -> bool {
        const auto& output_ps = output.get_partial_shape();
        return (output_ps.rank().is_static()) && (output_ps.rank().get_length() >= 2);
    };

    auto input_m = any_input(all_of({has_real_not_quantized_type, has_at_least_2d_shape, has_static_dim(1)}));

    auto pre_mvn_shape_const_m = wrap_type<ov::op::v0::Constant>(all_of({rank_equals(1), has_static_dim(0)}));
    auto pre_mvn_reshape_m =
        wrap_type<ov::op::v1::Reshape>({input_m, pre_mvn_shape_const_m},
                                       all_of({has_real_not_quantized_type, rank_equals(3), has_static_dim(1)}));

    auto mvn_reduction_axes_const_m = wrap_type<ov::op::v0::Constant>(all_of({rank_equals(1), has_static_dim(0)}));
    auto mvn_m = wrap_type<ov::op::v6::MVN>({pre_mvn_reshape_m, mvn_reduction_axes_const_m});

    auto instance_norm_gamma_m = any_input(all_of({has_real_not_quantized_type, has_static_shape()}));
    auto instance_norm_opt_gamma_m = optional<ov::op::v1::Multiply>({mvn_m, instance_norm_gamma_m});

    auto instance_norm_beta_m = any_input(all_of({has_real_not_quantized_type, has_static_shape()}));
    auto instance_norm_opt_gamma_opt_beta_m =
        optional<ov::op::v1::Add>({instance_norm_opt_gamma_m, instance_norm_beta_m});

    auto post_instance_norm_shape_m = any_input(all_of({rank_equals(1), has_static_dim(0)}));
    auto post_instance_norm_reshape_m =
        wrap_type<ov::op::v1::Reshape>({instance_norm_opt_gamma_opt_beta_m, post_instance_norm_shape_m},
                                       all_of({has_real_not_quantized_type, has_at_least_2d_shape, has_static_dim(1)}));

    auto group_norm_gamma_m = any_input(all_of({has_real_not_quantized_type, has_static_shape()}));
    auto group_norm_gamma_multiply_m =
        wrap_type<ov::op::v1::Multiply>({post_instance_norm_reshape_m, group_norm_gamma_m});

    auto group_norm_beta_m = any_input(all_of({has_real_not_quantized_type, has_static_shape()}));
    auto group_norm_beta_add_m = wrap_type<ov::op::v1::Add>({group_norm_gamma_multiply_m, group_norm_beta_m});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        const auto& input = pattern_map.at(input_m);
        const auto& input_ps = input.get_partial_shape();

        const auto& T = input.get_element_type();

        const auto& pre_mvn_reshape_out_ps = pattern_map.at(pre_mvn_reshape_m).get_partial_shape();

        const size_t num_channels = static_cast<size_t>(input_ps[1].get_max_length());
        const size_t num_groups = static_cast<size_t>(pre_mvn_reshape_out_ps[1].get_max_length());

        // we expect to reshape input in a way that would merge all spatial dimensions
        // but leave batch and channel dimensions untouched
        const auto& pre_mvn_shape = pattern_map.at(pre_mvn_shape_const_m);
        const auto& pre_mvn_shape_const =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(pre_mvn_shape_const_m).get_node_shared_ptr());
        const auto& pre_mvn_shape_out_ps = pre_mvn_shape.get_shape();
        if (pre_mvn_shape_out_ps[0] != 3)
            return false;

        auto pre_mvn_shape_vals_correct = [](const std::vector<int64_t>& pre_mvn_shape_vals,
                                             const ov::PartialShape& input_ps,
                                             const ov::Dimension::value_type num_groups) -> bool {
            bool res = true;
            if (input_ps[0].is_dynamic()) {
                if (pre_mvn_shape_vals[0] != 0ll)
                    res = false;
            } else {
                if ((pre_mvn_shape_vals[0] != 0ll) &&
                    (pre_mvn_shape_vals[0] != static_cast<long long>(input_ps[0].get_max_length())))
                    res = false;
            }
            if ((pre_mvn_shape_vals[1] != 0ll) && (pre_mvn_shape_vals[1] != static_cast<long long>(num_groups)))
                res = false;
            if (pre_mvn_shape_vals[2] != -1ll)
                res = false;
            return res;
        };

        if (!pre_mvn_shape_vals_correct(pre_mvn_shape_const->cast_vector<int64_t>(), input_ps, num_groups))
            return false;

        // number of channels has to be divisible by number of groups
        if (num_channels % num_groups != 0)
            return false;

        // first dimension of MVN input (batch_size) has to be the same
        // as in pattern input
        if (input_ps[0].get_max_length() != pre_mvn_reshape_out_ps[0].get_max_length())
            return false;

        // we expect to execute normalization over last dimension of MVN input
        const auto& mvn_reduction_axes = pattern_map.at(mvn_reduction_axes_const_m);
        const auto& mvn_reduction_axes_const =
            ov::as_type_ptr<ov::op::v0::Constant>(mvn_reduction_axes.get_node_shared_ptr());
        const auto& mvn_reduction_axes_out_shape = mvn_reduction_axes.get_shape();
        if (mvn_reduction_axes_out_shape[0] != 1)
            return false;

        auto mvn_reduction_axes_correct = [](const std::vector<int64_t>& mvn_reduction_axes) -> bool {
            bool res = true;
            if ((mvn_reduction_axes[0] != 2ll) && (mvn_reduction_axes[0] != -1ll))
                return false;
            return res;
        };

        if (!mvn_reduction_axes_correct(mvn_reduction_axes_const->cast_vector<int64_t>()))
            return false;

        const auto& post_instance_norm_reshape_out_ps =
            pattern_map.at(post_instance_norm_reshape_m).get_partial_shape();
        // post instance norm shape has to be same as in pattern input
        if (post_instance_norm_reshape_out_ps != input_ps)
            return false;

        const auto& group_norm_gamma = pattern_map.at(group_norm_gamma_m);
        if (group_norm_gamma.get_element_type() != T)
            return false;
        if (ov::shape_size(group_norm_gamma.get_shape()) != num_channels)
            return false;

        const auto& group_norm_beta = pattern_map.at(group_norm_beta_m);
        if (group_norm_beta.get_element_type() != T)
            return false;
        if (ov::shape_size(group_norm_beta.get_shape()) != num_channels)
            return false;

        ov::NodeVector nodes;

        std::shared_ptr<ov::Node> group_norm_gamma_1d_m = std::make_shared<ov::op::v0::Squeeze>(group_norm_gamma);
        nodes.push_back(group_norm_gamma_1d_m);
        const auto& group_norm_gamma_1d_out_ps = group_norm_gamma_1d_m->get_output_partial_shape(0);

        auto expected_param_shape = ov::PartialShape({static_cast<ov::Dimension>(num_channels)});
        if (group_norm_gamma_1d_out_ps != expected_param_shape)
            return false;

        std::shared_ptr<ov::Node> group_norm_beta_1d_m = std::make_shared<ov::op::v0::Squeeze>(group_norm_beta);
        nodes.push_back(group_norm_beta_1d_m);
        const auto& group_norm_beta_1d_out_ps = group_norm_beta_1d_m->get_output_partial_shape(0);

        if (group_norm_beta_1d_out_ps != expected_param_shape)
            return false;

        auto gather_axis_const_m = op::v0::Constant::create(element::i64, Shape{1}, {0});
        nodes.push_back(gather_axis_const_m);
        auto gather_indices_vals = std::vector<int64_t>();
        for (auto i = 0ull; i < num_groups; i++)
            gather_indices_vals.insert(gather_indices_vals.end(), num_channels / num_groups, i);
        auto gather_indices_const_m = op::v0::Constant::create(element::i64, Shape{num_channels}, gather_indices_vals);
        nodes.push_back(gather_indices_const_m);

        if (pattern_map.count(instance_norm_beta_m) > 0) {
            const auto& instance_norm_beta = pattern_map.at(instance_norm_beta_m);
            if (instance_norm_beta.get_element_type() != T)
                return false;
            if (ov::shape_size(instance_norm_beta.get_shape()) != num_groups)
                return false;

            // ensure that instance_norm_beta will have shape compatible
            // with group_norm parameters, i.e. 1D vector of shape (num_channels)
            std::shared_ptr<ov::Node> instance_norm_beta_1d_m = nullptr;
            if (ov::shape_size(instance_norm_beta.get_shape()) == 1) {
                auto shape_1d_const_m = op::v0::Constant::create(element::i64, Shape{1}, {1});
                nodes.push_back(shape_1d_const_m);
                instance_norm_beta_1d_m =
                    std::make_shared<ov::op::v1::Reshape>(instance_norm_beta, shape_1d_const_m, true);
                nodes.push_back(instance_norm_beta_1d_m);
            } else {
                instance_norm_beta_1d_m = std::make_shared<ov::op::v0::Squeeze>(instance_norm_beta);
                nodes.push_back(instance_norm_beta_1d_m);
            }

            instance_norm_beta_1d_m = std::make_shared<ov::op::v8::Gather>(instance_norm_beta_1d_m,
                                                                           gather_indices_const_m,
                                                                           gather_axis_const_m);
            nodes.push_back(instance_norm_beta_1d_m);

            const auto& instance_norm_beta_1d_ps = instance_norm_beta_1d_m->get_output_partial_shape(0);
            if (instance_norm_beta_1d_ps != expected_param_shape)
                return false;

            // group_norm_beta = group_norm_gamma * instance_norm_beta + group_norm_beta
            auto group_norm_beta_corr_multiply_m =
                std::make_shared<ov::op::v1::Multiply>(group_norm_gamma_1d_m, instance_norm_beta_1d_m);
            nodes.push_back(group_norm_beta_corr_multiply_m);
            group_norm_beta_1d_m =
                std::make_shared<ov::op::v1::Add>(group_norm_beta_corr_multiply_m, group_norm_beta_1d_m);
            nodes.push_back(group_norm_beta_1d_m);
        }

        if (pattern_map.count(instance_norm_gamma_m) > 0) {
            const auto& instance_norm_gamma = pattern_map.at(instance_norm_gamma_m);
            if (instance_norm_gamma.get_element_type() != T)
                return false;
            if (ov::shape_size(instance_norm_gamma.get_shape()) != num_groups)
                return false;

            // ensure that instance_norm_gamma will have shape compatible
            // with group_norm parameters, i.e. 1D vector of shape (num_channels)
            std::shared_ptr<ov::Node> instance_norm_gamma_1d_m = nullptr;
            if (ov::shape_size(instance_norm_gamma.get_shape()) == 1) {
                auto shape_1d_const_m = op::v0::Constant::create(element::i64, Shape{1}, {1});
                nodes.push_back(shape_1d_const_m);
                instance_norm_gamma_1d_m =
                    std::make_shared<ov::op::v1::Reshape>(instance_norm_gamma, shape_1d_const_m, true);
                nodes.push_back(instance_norm_gamma_1d_m);
            } else {
                instance_norm_gamma_1d_m = std::make_shared<ov::op::v0::Squeeze>(instance_norm_gamma);
                nodes.push_back(instance_norm_gamma_1d_m);
            }

            instance_norm_gamma_1d_m = std::make_shared<ov::op::v8::Gather>(instance_norm_gamma_1d_m,
                                                                            gather_indices_const_m,
                                                                            gather_axis_const_m);
            nodes.push_back(instance_norm_gamma_1d_m);

            const auto& instance_norm_gamma_1d_ps = instance_norm_gamma_1d_m->get_output_partial_shape(0);
            if (instance_norm_gamma_1d_ps != expected_param_shape)
                return false;

            // group_norm_gamma *= instance_norm_gamma
            group_norm_gamma_1d_m =
                std::make_shared<ov::op::v1::Multiply>(group_norm_gamma_1d_m, instance_norm_gamma_1d_m);
            nodes.push_back(group_norm_gamma_1d_m);
        }

        // we need to cast mvn to MVN layer type in order to read actual epsilon value
        const auto& mvn_out = pattern_map.at(mvn_m);
        const auto& mvn = ov::as_type_ptr<ov::op::v6::MVN>(mvn_out.get_node_shared_ptr());
        const auto& epsilon = mvn->get_eps();

        // reuse original friendly names for gamma and beta inputs
        group_norm_gamma_1d_m->set_friendly_name(group_norm_gamma_m->get_friendly_name());
        group_norm_beta_1d_m->set_friendly_name(group_norm_beta_m->get_friendly_name());

        // we can finally create GroupNormalization op
        std::shared_ptr<ov::Node> group_norm = std::make_shared<ov::op::v12::GroupNormalization>(input,
                                                                                                 group_norm_gamma_1d_m,
                                                                                                 group_norm_beta_1d_m,
                                                                                                 num_groups,
                                                                                                 epsilon);
        nodes.push_back(group_norm);

        // and do actual graph substitution
        group_norm->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), nodes);
        ov::replace_node(m.get_match_root(), group_norm);
        return true;
    };

    auto m = std::make_shared<Matcher>(group_norm_beta_add_m, matcher_name);
    this->register_matcher(m, callback);
}
