// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_convolution_to_matmul.hpp"

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ConvertConvolutionToMatMul::ConvertConvolutionToMatMul(const element::TypeVector& supported_precisions,
                                                                 const element::TypeVector& unsupported_precisions) {
    MATCHER_SCOPE(ConvertConvolutionToMatMul);

    auto final_precisions = supported_precisions;
    if (!unsupported_precisions.empty()) {
        final_precisions.erase(std::remove_if(final_precisions.begin(),
                                              final_precisions.end(),
                                              [&](const ov::element::Type& type) {
                                                  return std::find(unsupported_precisions.begin(),
                                                                   unsupported_precisions.end(),
                                                                   type) != unsupported_precisions.end();
                                              }),
                               final_precisions.end());
    }

    auto check_precision = [](const ov::element::TypeVector& precisions) -> ov::pass::pattern::op::Predicate {
        return ov::pass::pattern::op::Predicate([=](const Output<Node>& output) -> bool {
            return std::find(precisions.begin(), precisions.end(), output.get_element_type()) != precisions.end();
        });
    };

    auto weights = pattern::any_input(check_precision(final_precisions));
    auto weights_convert = pattern::wrap_type<ov::op::v0::Convert>({weights}, pattern::consumers_count(1));
    auto zp = pattern::any_input();
    auto zp_convert = pattern::optional<ov::op::v0::Convert>(zp);
    auto zp_reshape = pattern::optional<ov::op::v1::Reshape, ov::op::v0::Unsqueeze>({zp_convert, pattern::any_input()});
    auto weights_sub = pattern::optional<ov::op::v1::Subtract>({weights_convert, zp_reshape});
    auto scale = pattern::any_input();
    auto scale_convert = pattern::optional<ov::op::v0::Convert>(scale);
    auto scale_reshape =
        pattern::optional<ov::op::v1::Reshape, ov::op::v0::Unsqueeze>({scale_convert, pattern::any_input()});
    auto weights_sub_multiply = pattern::wrap_type<ov::op::v1::Multiply>({weights_sub, scale_reshape});
    auto weights_sub_multiply_reshape =
        pattern::optional<ov::op::v1::Reshape>({weights_sub_multiply, pattern::any_input()},
                                               pattern::shape_matches("[hidden_out, hidden_in, 1, 1]"));
    auto conv_input_1 = pattern::any_input(pattern::shape_matches("[?, ?, 1, 1]"));
    auto conv_input_2 = pattern::any_input(pattern::shape_matches("[1, ?, 1, ?]"));
    auto conv_input_3 = pattern::any_input(pattern::shape_matches("[1, ?, ?, 1]"));
    auto conv_input =
        std::make_shared<ov::pass::pattern::op::Or>(OutputVector{conv_input_1, conv_input_2, conv_input_3});

    auto conv_pattern = pattern::wrap_type<ov::op::v1::Convolution>({conv_input, weights_sub_multiply_reshape},
                                                                    {{"auto_pad", "explicit"},
                                                                     {"dilations", std::vector<int64_t>{1, 1}},
                                                                     {"strides", std::vector<int64_t>{1, 1}},
                                                                     {"pads_begin", std::vector<int64_t>{0, 0}},
                                                                     {"pads_end", std::vector<int64_t>{0, 0}}});

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto conv_node = ov::as_type_ptr<ov::op::v1::Convolution>(m.get_match_root());
        if (!conv_node) {
            return false;
        }

        auto weights = conv_node->input_value(1);
        const auto& weights_partial_shape = weights.get_partial_shape();
        if (weights_partial_shape.is_dynamic() || weights_partial_shape.size() != 4 || weights_partial_shape[2] != 1 ||
            weights_partial_shape[3] != 1) {
            return false;
        }
        auto hidden_out = weights_partial_shape[0].get_length();
        auto hidden_in = weights_partial_shape[1].get_length();

        const auto& input_shape = conv_node->get_input_partial_shape(0);
        std::vector<int64_t> input_transpose_order, output_transpose_order;
        if (input_shape[0] == 1 && input_shape[2] == 1) {
            input_transpose_order = {0, 2, 3, 1};  // [1, hidden_in, 1, seq_len] -> [1, 1, seq_len, hidden_in]
            output_transpose_order = {0, 3, 1, 2};
        } else if (input_shape[2] == 1 && input_shape[3] == 1) {
            input_transpose_order = {2, 3, 0, 1};  // [seq_len, hidden_in, 1, 1] -> [1, 1, seq_len, hidden_in]
            output_transpose_order = {2, 3, 0, 1};
        } else if (input_shape[0] == 1 && input_shape[3] == 1) {
            input_transpose_order = {0, 3, 2, 1};  // [1, hidden_in, seq_len, 1] -> [1, 1, seq_len, hidden_in]
            output_transpose_order = {0, 3, 2, 1};
        } else {
            return false;
        }

        // Reshape weights from 1x1 kernel [hidden_out, hidden_in, 1, 1] to matmul var b [hidden_out, hidden_in]
        auto reshape_weights_pattern =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                   ov::Shape{2},
                                                   std::vector<int64_t>{hidden_out, hidden_in});
        auto reshape_weights = std::make_shared<ov::op::v1::Reshape>(weights, reshape_weights_pattern, false);

        // Transpose input to [1, 1, seq_len, hidden_in]
        auto input = conv_node->input_value(0);
        auto input_transpose_const =
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, input_transpose_order);
        auto transpose_input = std::make_shared<ov::op::v1::Transpose>(input, input_transpose_const);

        // MatMul: [1, 1, seq_len, hidden_in] x [hidden_out, hidden_in]^T => [1, 1, seq_len, hidden_out]
        auto matmul = std::make_shared<ov::op::v0::MatMul>(transpose_input, reshape_weights, false, true);

        // Transpose output back to the original layout
        auto output_transpose_const =
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, output_transpose_order);
        auto final_node = std::make_shared<ov::op::v1::Transpose>(matmul, output_transpose_const);

        final_node->set_friendly_name(conv_node->get_friendly_name());
        ov::copy_runtime_info(conv_node,
                              {reshape_weights_pattern, reshape_weights, transpose_input, matmul, final_node});
        ov::replace_node(conv_node, final_node);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(conv_pattern, matcher_name);
    this->register_matcher(m, callback);
}
