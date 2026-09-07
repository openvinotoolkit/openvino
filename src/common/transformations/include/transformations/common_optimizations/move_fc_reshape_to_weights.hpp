// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief MoveFCReshapeToWeights is applied to FC with compressed 3D weights (e.g. u8/i4).
 * It moves the 3D->2D Reshape on the weights path into the constant subgraph by squeezing
 * the leading size-1 dimension out of every constant node, allowing ConstantFolding to
 * eliminate the Reshape entirely.
 *
 *  Weights(3D)   ZP_const(3D)       Weights(2D)   ZP_const(2D)
 *      |             /                   |             /
 *   Convert    ZP_convert(opt)        Convert    ZP_convert(opt)
 *      |      /                           |      /
 *   Subtract(opt)             ===>     Subtract(opt)
 *      |      Scale_const(3D)             |      Scale_const(2D)
 *       \     /                            \     /
 *       Multiply                           Multiply
 *          |                                  |
 *       Reshape(2D)                           |
 *          |                                  |
 *  Data  Transpose(opt)            Data  Transpose(opt)
 *     \  /                              \  /
 *   FullyConnected                    FullyConnected
 */
template <typename FCType>
class MoveFCReshapeToWeights : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MoveFCReshapeToWeights");
    MoveFCReshapeToWeights() {
        using namespace ov::pass::pattern;

        auto weights_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
        auto convert_m = wrap_type<ov::op::v0::Convert>({weights_m}, consumers_count(1));

        auto sub_const_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
        auto subtract_wo_convert_m = wrap_type<ov::op::v1::Subtract>({convert_m, sub_const_m}, consumers_count(1));
        auto sub_convert_m = wrap_type<ov::op::v0::Convert>({sub_const_m}, consumers_count(1));
        auto subtract_w_convert_m = wrap_type<ov::op::v1::Subtract>({convert_m, sub_convert_m}, consumers_count(1));
        auto subtract_m = std::make_shared<op::Or>(OutputVector{subtract_wo_convert_m, subtract_w_convert_m});

        auto one_consumer_rank3 = [](const ov::Output<ov::Node>& out) {
            return consumers_count(1)(out) && rank_equals(3)(out);
        };
        auto mul_const_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
        auto mul_with_sub_m = wrap_type<ov::op::v1::Multiply>({subtract_m, mul_const_m}, one_consumer_rank3);
        auto mul_no_sub_m = wrap_type<ov::op::v1::Multiply>({convert_m, mul_const_m}, one_consumer_rank3);
        auto mul_m = std::make_shared<op::Or>(OutputVector{mul_with_sub_m, mul_no_sub_m});

        auto one_consumer_rank2 = [](const ov::Output<ov::Node>& out) {
            return consumers_count(1)(out) && rank_equals(2)(out);
        };
        auto reshape_const_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
        auto reshape_m = wrap_type<ov::op::v1::Reshape>({mul_m, reshape_const_m}, one_consumer_rank2);

        auto transpose_const_m = wrap_type<ov::op::v0::Constant>();
        auto transpose_m = wrap_type<ov::op::v1::Transpose>({reshape_m, transpose_const_m});
        auto weights_input_m = std::make_shared<op::Or>(OutputVector{reshape_m, transpose_m});

        auto data_m = any_input();
        auto bias_m = any_input();
        auto fully_connected_m = wrap_type<FCType>({data_m, weights_input_m, bias_m});

        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
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

            // Accepts shapes that are exactly {1, out_channels, 1} (with the out_channels
            // dimension at the correct index), or any shape that is all-ones / shorter rank
            // (broadcast-compatible scalars).
            auto check_decompression_shape = [&](const std::shared_ptr<ov::Node>& node) {
                ov::Shape expected_shape(3, 1);
                const size_t out_channels_idx = with_transpose ? 2 : 1;
                expected_shape[out_channels_idx] = fc_input_shape[0];
                const auto& node_shape = node->get_output_shape(0);
                if (node_shape.size() > expected_shape.size()) {
                    return false;
                }
                const auto offset = expected_shape.size() - node_shape.size();
                return std::equal(node_shape.begin(), node_shape.end(), expected_shape.begin() + offset) ||
                       std::all_of(node_shape.cbegin(), node_shape.cend(), [](ov::Shape::value_type d) {
                           return d == 1;
                       });
            };

            const auto mul = reshape->get_input_node_shared_ptr(0);
            if (!check_decompression_shape(mul->get_input_node_shared_ptr(1)))
                return false;

            const auto mul_parent = mul->get_input_node_shared_ptr(0);
            const bool with_subtract = ov::is_type<ov::op::v1::Subtract>(mul_parent);
            if (with_subtract && !check_decompression_shape(mul_parent->get_input_node_shared_ptr(1)))
                return false;

            const auto convert = with_subtract ? mul_parent->get_input_node_shared_ptr(0) : mul_parent;
            const auto weights = convert->get_input_node_shared_ptr(0);
            ov::Shape expected_weights_shape(3, 1);
            expected_weights_shape[1] = fc_input_shape[with_transpose ? 1 : 0];
            expected_weights_shape[2] = fc_input_shape[with_transpose ? 0 : 1];
            if (weights->get_output_shape(0) != expected_weights_shape)
                return false;

            // Only squeeze when the constant has exactly one extra leading dimension.
            auto squeeze_constant = [&](const std::shared_ptr<ov::Node>& node) {
                const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
                OPENVINO_ASSERT(constant, "Expected Constant node, got: ", node->get_type_name());
                auto shape = constant->get_shape();
                if (shape.size() == fc_input_shape.size() + 1) {
                    shape.erase(shape.begin());
                    const auto new_constant = std::make_shared<ov::op::v0::Constant>(*constant, shape);
                    ov::replace_node(constant, new_constant);
                    ov::copy_runtime_info(constant, new_constant);
                    new_constant->set_friendly_name(constant->get_friendly_name());
                }
            };

            // Remove the 3D->2D reshape by squeezing constants in the weights subgraph.
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

        auto matcher = std::make_shared<ov::pass::pattern::Matcher>(fully_connected_m);
        this->register_matcher(matcher, callback);
    }
};

}  // namespace pass
}  // namespace ov
