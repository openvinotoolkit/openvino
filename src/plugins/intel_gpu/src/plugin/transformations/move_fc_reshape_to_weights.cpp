// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "move_fc_reshape_to_weights.hpp"
#include "intel_gpu/op/fully_connected.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"

#include "openvino/op/matmul.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/reshape.hpp"

namespace ov::intel_gpu {

MoveFCReshapeToWeights::MoveFCReshapeToWeights() {
    using namespace ov::pass::pattern;

    auto weights_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto convert_m = wrap_type<ov::op::v0::Convert>({weights_m});

    auto sub_const_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto subtract_m = wrap_type<ov::op::v1::Subtract>({convert_m, sub_const_m});

    auto mul_const_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto mul_with_sub_m = wrap_type<ov::op::v1::Multiply>({subtract_m, mul_const_m}, rank_equals(3));
    auto mul_no_sub_m = wrap_type<ov::op::v1::Multiply>({convert_m, mul_const_m}, rank_equals(3));
    auto mul_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{mul_with_sub_m, mul_no_sub_m});

    auto one_consumer_rank_2 = [](const ov::Output<ov::Node>& out) {
        return consumers_count(1)(out) && rank_equals(2)(out);
    };
    auto reshape_const_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto reshape_m = wrap_type<ov::op::v1::Reshape>({mul_m, reshape_const_m}, one_consumer_rank_2);

    auto transpose_const_m = wrap_type<ov::op::v0::Constant>();
    auto transpose_m = wrap_type<ov::op::v1::Transpose>({reshape_m, transpose_const_m});
    auto weights_input_m = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{reshape_m, transpose_m});

    auto data_m = any_input();
    auto fully_connected_m = wrap_type<op::FullyConnected>({data_m, weights_input_m, any_input()});

    ov::matcher_pass_callback callback = [&](ov::pass::pattern::Matcher& m) {
        const auto fully_connected = m.get_match_root();
        const auto weights_path = fully_connected->get_input_node_shared_ptr(1);
        const bool with_transpose = ov::is_type<ov::op::v1::Transpose>(weights_path);
        if (with_transpose) {
            const auto transpose_const = ov::as_type_ptr<ov::op::v0::Constant>(weights_path->get_input_node_shared_ptr(1));
            if (transpose_const->cast_vector<int>() != std::vector<int>{1, 0}) {
                return false;
            }
        }

        const auto& fc_input_shape = fully_connected->get_input_shape(1);
        const auto reshape = with_transpose ? weights_path->get_input_node_shared_ptr(0) : weights_path;

        auto check_decompression_const = [&](const std::shared_ptr<ov::Node>& node) {
            if (!ov::is_type<ov::op::v0::Constant>(node))
                return false;
            ov::Shape expected_shape(3, 1);
            const size_t out_channels_idx = with_transpose ? 2 : 1;
            expected_shape[out_channels_idx] = fc_input_shape[0];
            return node->get_output_shape(0) == expected_shape;
        };

        const auto mul = reshape->get_input_node_shared_ptr(0);
        if (!check_decompression_const(mul->get_input_node_shared_ptr(1)))
            return false;
        const auto mul_parent = mul->get_input_node_shared_ptr(0);
        const bool with_subtract = ov::is_type<ov::op::v1::Subtract>(mul_parent);
        if (with_subtract && !check_decompression_const(mul_parent->get_input_node_shared_ptr(1)))
            return false;

        const auto convert = with_subtract ? mul_parent->get_input_node_shared_ptr(0) : mul_parent;
        const auto weights = convert->get_input_node_shared_ptr(0);
        ov::Shape expected_weights_shape(3, 1);
        expected_weights_shape[1] = fc_input_shape[with_transpose ? 1 : 0];
        expected_weights_shape[2] = fc_input_shape[with_transpose ? 0 : 1];
        if (weights->get_output_shape(0) != expected_weights_shape)
            return false;

        auto squeeze_constant = [](const std::shared_ptr<ov::Node>& node) {
            const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
            auto shape = constant->get_shape();
            shape.erase(shape.begin());
            const auto new_constant = std::make_shared<ov::op::v0::Constant>(*constant, shape);
            ov::replace_node(constant, new_constant);
            ov::copy_runtime_info(constant, new_constant);
            new_constant->set_friendly_name(constant->get_friendly_name());
        };

        // We can remove 3D->2D reshape if we manually reshape all constants in the weights subgraph
        ov::replace_output_update_name(reshape->output(0), reshape->input_value(0));
        squeeze_constant(mul->get_input_node_shared_ptr(1));
        squeeze_constant(weights);
        if (with_subtract)
            squeeze_constant(mul_parent->get_input_node_shared_ptr(1));
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fully_connected_m);
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
