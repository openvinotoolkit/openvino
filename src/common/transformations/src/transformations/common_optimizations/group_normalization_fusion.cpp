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

template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
bool pre_mvn_shape_vals_correct(const std::shared_ptr<ov::op::v0::Constant>& pre_mvn_shape_const,
                                const ov::PartialShape& input_ps,
                                const ov::Dimension::value_type num_groups) {
    bool res = true;
    std::vector<T> pre_mvn_shape_vals = pre_mvn_shape_const->get_vector<T>();
    if (input_ps[0].is_dynamic()) {
        if (pre_mvn_shape_vals[0] != 0)
            res = false;
    } else {
        if ((pre_mvn_shape_vals[0] != 0) && (pre_mvn_shape_vals[0] != input_ps[0].get_max_length()))
            res = false;
    }
    if ((pre_mvn_shape_vals[1] != 0) && (pre_mvn_shape_vals[1] != num_groups))
        res = false;
    if (pre_mvn_shape_vals[2] != -1)
        res = false;
    return res;
}

template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
bool mvn_reduction_axes_correct(const std::shared_ptr<ov::op::v0::Constant>& mvn_reduction_axes_const) {
    bool res = true;
    std::vector<T> mvn_reduce_axes = mvn_reduction_axes_const->get_vector<T>();
    if ((mvn_reduce_axes[0] != 2) && (mvn_reduce_axes[0] != -1))
        return false;
    return res;
}

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

        const auto& num_channels = input_ps[1].get_max_length();
        const auto& num_groups = pre_mvn_reshape_out_ps[1].get_max_length();

        // we expect to reshape input in a way that would merge all spatial dimensions
        // but leave batch and channel dimensions untouched
        const auto& pre_mvn_shape = pattern_map.at(pre_mvn_shape_const_m);
        const auto& pre_mvn_shape_const =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(pre_mvn_shape_const_m).get_node_shared_ptr());
        const auto& pre_mvn_shape_out_ps = pre_mvn_shape.get_shape();
        if (pre_mvn_shape_out_ps[0] != 3)
            return false;
        switch (pre_mvn_shape_const->get_element_type()) {
        case ov::element::i8:
            if (!pre_mvn_shape_vals_correct<int8_t>(pre_mvn_shape_const, input_ps, num_groups))
                return false;
            break;
        case ov::element::i16:
            if (!pre_mvn_shape_vals_correct<int16_t>(pre_mvn_shape_const, input_ps, num_groups))
                return false;
            break;
        case ov::element::i32:
            if (!pre_mvn_shape_vals_correct<int32_t>(pre_mvn_shape_const, input_ps, num_groups))
                return false;
            break;
        case ov::element::i64:
            if (!pre_mvn_shape_vals_correct<int64_t>(pre_mvn_shape_const, input_ps, num_groups))
                return false;
            break;
        case ov::element::u8:
            if (!pre_mvn_shape_vals_correct<uint8_t>(pre_mvn_shape_const, input_ps, num_groups))
                return false;
            break;
        case ov::element::u16:
            if (!pre_mvn_shape_vals_correct<uint16_t>(pre_mvn_shape_const, input_ps, num_groups))
                return false;
            break;
        case ov::element::u32:
            if (!pre_mvn_shape_vals_correct<uint32_t>(pre_mvn_shape_const, input_ps, num_groups))
                return false;
            break;
        case ov::element::u64:
            if (!pre_mvn_shape_vals_correct<uint64_t>(pre_mvn_shape_const, input_ps, num_groups))
                return false;
            break;
        default:
            return false;
        }

        // number of channels has to be divisible by number of groups
        if (num_channels % num_groups != 0)
            return false;
        auto channels_to_groups_ratio = num_channels / num_groups;

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
        switch (mvn_reduction_axes_const->get_element_type()) {
        case ov::element::i32:
            mvn_reduction_axes_correct<int32_t>(mvn_reduction_axes_const);
            break;
        case ov::element::i64:
            mvn_reduction_axes_correct<int64_t>(mvn_reduction_axes_const);
            break;
        default:
            break;
        }

        const auto& post_instance_norm_reshape_out_ps =
            pattern_map.at(post_instance_norm_reshape_m).get_partial_shape();
        // post instance norm shape has to be same as in pattern input
        if (post_instance_norm_reshape_out_ps != input_ps)
            return false;

        const auto& group_norm_gamma = pattern_map.at(group_norm_gamma_m);
        // group_norm_gamma has to share the same data type as
        // pattern input
        if (group_norm_gamma.get_element_type() != T)
            return false;

        // number of elements in group_norm_gamma must be equal to
        // number of channels
        if (ov::shape_size(group_norm_gamma.get_shape()) != num_channels)
            return false;

        const auto& group_norm_beta = pattern_map.at(group_norm_beta_m);

        // group_norm_beta has to share the same data type as
        // pattern input
        if (group_norm_beta.get_element_type() != T)
            return false;

        // number of elements in group_norm_beta must be equal to
        // number of channels
        if (ov::shape_size(group_norm_beta.get_shape()) != num_channels)
            return false;

        auto expected_param_shape = ov::PartialShape({num_channels});

        std::shared_ptr<ov::Node> group_norm_gamma_1d_m = std::make_shared<ov::op::v0::Squeeze>(group_norm_gamma);
        const auto& group_norm_gamma_1d_out_ps = group_norm_gamma_1d_m->get_output_partial_shape(0);

        if (group_norm_gamma_1d_out_ps != expected_param_shape)
            return false;

        std::shared_ptr<ov::Node> group_norm_beta_1d_m = std::make_shared<ov::op::v0::Squeeze>(group_norm_beta);
        const auto& group_norm_beta_1d_out_ps = group_norm_beta_1d_m->get_output_partial_shape(0);

        if (group_norm_beta_1d_out_ps != expected_param_shape)
            return false;

        auto gather_axis_const_m = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto gather_indices_vals = std::vector<int64_t>();
        for (auto i = 0; i < num_groups; i++)
            gather_indices_vals.insert(gather_indices_vals.end(), channels_to_groups_ratio, i);
        auto gather_indices_const_m =
            op::v0::Constant::create(element::i64, Shape{static_cast<size_t>(num_channels)}, gather_indices_vals);

        std::shared_ptr<ov::Node> instance_norm_beta_1d_m = nullptr;
        if (pattern_map.count(instance_norm_beta_m) > 0) {
            const auto& instance_norm_beta = pattern_map.at(instance_norm_beta_m);

            // instance_norm_beta has to share the same data type as
            // pattern input
            if (instance_norm_beta.get_element_type() != T)
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

            instance_norm_beta_1d_m = std::make_shared<ov::op::v8::Gather>(instance_norm_beta_1d_m,
                                                                           gather_indices_const_m,
                                                                           gather_axis_const_m);

            const auto& instance_norm_beta_1d_ps = instance_norm_beta_1d_m->get_output_partial_shape(0);
            if (instance_norm_beta_1d_ps != expected_param_shape)
                return false;

            // group_norm_beta = group_norm_gamma * instance_norm_beta + group_norm_beta
            auto group_norm_beta_corr_multiply_m =
                std::make_shared<ov::op::v1::Multiply>(group_norm_gamma_1d_m, instance_norm_beta_1d_m);
            group_norm_beta_1d_m =
                std::make_shared<ov::op::v1::Add>(group_norm_beta_corr_multiply_m, group_norm_beta_1d_m);
        }

        if (pattern_map.count(instance_norm_gamma_m) > 0) {
            const auto& instance_norm_gamma = pattern_map.at(instance_norm_gamma_m);

            // instance_norm_gamma has to share the same data type as
            // pattern input
            if (instance_norm_gamma.get_element_type() != T)
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

            instance_norm_gamma_1d_m = std::make_shared<ov::op::v8::Gather>(instance_norm_gamma_1d_m,
                                                                            gather_indices_const_m,
                                                                            gather_axis_const_m);
            const auto& instance_norm_gamma_1d_ps = instance_norm_gamma_1d_m->get_output_partial_shape(0);
            if (instance_norm_gamma_1d_ps != expected_param_shape)
                return false;

            // group_norm_gamma *= instance_norm_gamma
            group_norm_gamma_1d_m =
                std::make_shared<ov::op::v1::Multiply>(group_norm_gamma_1d_m, instance_norm_gamma_1d_m);
        }

        // we need to cast mvn to MVN layer type in order to read actual epsilon value
        const auto& mvn_out = pattern_map.at(mvn_m);
        const auto& mvn = ov::as_type_ptr<ov::op::v6::MVN>(mvn_out.get_node_shared_ptr());
        const auto& epsilon = mvn->get_eps();

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

    auto m = std::make_shared<Matcher>(group_norm_beta_add_m, matcher_name);
    this->register_matcher(m, callback);
}
