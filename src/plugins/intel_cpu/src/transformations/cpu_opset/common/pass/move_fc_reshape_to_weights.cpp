// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "move_fc_reshape_to_weights.hpp"

#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"
#include "ov_ops/fully_connected.hpp"

ov::intel_cpu::MoveFCReshapeToWeights::MoveFCReshapeToWeights() {
    MATCHER_SCOPE(MoveFCReshapeToWeights);
    using namespace ov::pass::pattern;
    auto weights_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto convert_m = wrap_type<ov::op::v0::Convert>({weights_m}, consumers_count(1));

    auto one_consumer_rank_equals = [](const ov::Dimension& expected_rank) {
        return [=](const ov::Output<ov::Node>& output) -> bool {
            return consumers_count(1)(output) && rank_equals(expected_rank)(output);
        };
    };

    auto sub_const_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto subtract_wo_convert_m = wrap_type<ov::op::v1::Subtract>({convert_m, sub_const_m}, consumers_count(1));
    auto sub_convert = wrap_type<ov::op::v0::Convert>({sub_const_m}, consumers_count(1));
    auto subtract_w_convert_m = wrap_type<ov::op::v1::Subtract>({convert_m, sub_convert}, consumers_count(1));
    auto subtract_m =
        std::make_shared<ov::pass::pattern::op::Or>(OutputVector{subtract_wo_convert_m, subtract_w_convert_m});

    auto mul_const_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto mul_with_sub_m = wrap_type<ov::op::v1::Multiply>({subtract_m, mul_const_m}, one_consumer_rank_equals(3));
    auto mul_no_sub_m = wrap_type<ov::op::v1::Multiply>({convert_m, mul_const_m}, one_consumer_rank_equals(3));
    auto mul_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{mul_with_sub_m, mul_no_sub_m});

    auto reshape_const_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto reshape_m = wrap_type<ov::op::v1::Reshape>({mul_m, reshape_const_m}, one_consumer_rank_equals(2));

    auto transpose_const_m = wrap_type<ov::op::v0::Constant>();
    auto transpose_m = wrap_type<ov::op::v1::Transpose>({reshape_m, transpose_const_m});
    auto weights_input_m = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{reshape_m, transpose_m});

    auto data_m = any_input();
    auto bias_m = any_input();
    auto fully_connected_m = wrap_type<ov::op::internal::FullyConnected>({data_m, weights_input_m, bias_m});

    ov::matcher_pass_callback callback = [&](ov::pass::pattern::Matcher& m) {
        const auto fully_connected = m.get_match_root();
        const auto weights_path = fully_connected->get_input_node_shared_ptr(1);
        const bool with_transpose = ov::is_type<ov::op::v1::Transpose>(weights_path);
        if (with_transpose) {
            const auto transpose_const =
                ov::as_type_ptr<ov::op::v0::Constant>(weights_path->get_input_node_shared_ptr(1));
            if (transpose_const->cast_vector<int>() != std::vector<int>{1, 0}) {
                return false;
            }
        }

        const auto& fc_input_shape = fully_connected->get_input_shape(1);
        const auto reshape = with_transpose ? weights_path->get_input_node_shared_ptr(0) : weights_path;

        auto check_decompression_shape = [&](const std::shared_ptr<ov::Node>& node) {
            ov::Shape expected_shape(3, 1);
            const size_t out_channels_idx = with_transpose ? 2 : 1;
            expected_shape[out_channels_idx] = fc_input_shape[0];
            const auto& node_shape = node->get_output_shape(0);
            if (node_shape.size() > expected_shape.size()) {
                return false;
            }

            const auto comparison_start_pos = expected_shape.size() - node_shape.size();
            return std::equal(node_shape.begin(), node_shape.end(), expected_shape.begin() + comparison_start_pos) ||
                   std::all_of(node_shape.cbegin(), node_shape.cend(), [](int dim) {
                       return dim == 1;
                   });
        };

        const auto mul = reshape->get_input_node_shared_ptr(0);
        if (!check_decompression_shape(mul->get_input_node_shared_ptr(1))) {
            return false;
        }
        const auto mul_parent = mul->get_input_node_shared_ptr(0);
        const bool with_subtract = ov::is_type<ov::op::v1::Subtract>(mul_parent);
        if (with_subtract && !check_decompression_shape(mul_parent->get_input_node_shared_ptr(1))) {
            return false;
        }

        const auto convert = with_subtract ? mul_parent->get_input_node_shared_ptr(0) : mul_parent;
        const auto weights = convert->get_input_node_shared_ptr(0);
        ov::Shape expected_weights_shape(3, 1);
        expected_weights_shape[1] = fc_input_shape[with_transpose ? 1 : 0];
        expected_weights_shape[2] = fc_input_shape[with_transpose ? 0 : 1];
        if (weights->get_output_shape(0) != expected_weights_shape) {
            return false;
        }

        auto squeeze_constant = [&](const std::shared_ptr<ov::Node>& node) {
            const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
            OPENVINO_ASSERT(constant, "squeeze_constant is called for non constant node");
            auto shape = constant->get_shape();
            if (shape.size() - fc_input_shape.size() == 1) {
                shape.erase(shape.begin());
                const auto new_constant = std::make_shared<ov::op::v0::Constant>(*constant, shape);
                ov::replace_node(constant, new_constant);
                ov::copy_runtime_info(constant, new_constant);
                new_constant->set_friendly_name(constant->get_friendly_name());
            }
        };

        // We can remove 3D->2D reshape if we manually reshape all constants in the weights subgraph
        ov::replace_output_update_name(reshape->output(0), reshape->input_value(0));
        squeeze_constant(mul->get_input_node_shared_ptr(1));
        squeeze_constant(weights);
        if (with_subtract) {
            auto sub_const = mul_parent->get_input_node_shared_ptr(1);
            if (ov::is_type<ov::op::v0::Convert>(sub_const)) {
                sub_const = sub_const->get_input_node_shared_ptr(0);
            }
            squeeze_constant(sub_const);
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fully_connected_m, matcher_name);
    this->register_matcher(m, callback);
}
