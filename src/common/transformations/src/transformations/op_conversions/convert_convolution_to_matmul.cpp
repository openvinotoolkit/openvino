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

ov::pass::ConvertConvolutionToMatMul::ConvertConvolutionToMatMul() {
    MATCHER_SCOPE(ConvertConvolutionToMatMul);

    auto weights = pattern::any_input();
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
        pattern::optional<ov::op::v1::Reshape>({weights_sub_multiply, pattern::any_input()});

    auto conv_pattern = pattern::wrap_type<ov::op::v1::Convolution>(
        {pattern::any_input(), weights_sub_multiply_reshape},
        [](const Output<Node>& output) -> bool {
            auto conv_node = ov::as_type_ptr<ov::op::v1::Convolution>(output.get_node_shared_ptr());
            if (!conv_node) {
                return false;
            }

            // weights should be static 1x1 kernel, [hidden_out, hidden_in, 1, 1]
            const auto& weights_shape = conv_node->get_input_partial_shape(1);
            if (weights_shape.is_dynamic()) {
                return false;
            }
            if (weights_shape.size() != 4 || weights_shape[2] != 1 || weights_shape[3] != 1) {
                return false;
            }

            // input should met: [seq_len, hidden_in, 1, 1], [1, hidden_in, 1, seq_len] or [1, hidden_in, seq_len, 1]
            const auto& input_shape = conv_node->get_input_partial_shape(0);
            if (input_shape.rank().get_length() != 4) {
                return false;
            }
            const bool is_supported_shape = (input_shape[2] == 1 && input_shape[3] == 1) ||
                                            (input_shape[0] == 1 && input_shape[2] == 1) ||
                                            (input_shape[0] == 1 && input_shape[3] == 1);
            if (!is_supported_shape) {
                return false;
            }

            // stride/dilation should be 1, pad should be 0
            return conv_node->get_strides() == ov::Strides{1, 1} && conv_node->get_dilations() == ov::Strides{1, 1} &&
                   conv_node->get_pads_begin() == ov::CoordinateDiff{0, 0} &&
                   conv_node->get_pads_end() == ov::CoordinateDiff{0, 0};
        });

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto conv_node = ov::as_type_ptr<ov::op::v1::Convolution>(m.get_match_root());
        if (!conv_node) {
            return false;
        }

        auto weights = conv_node->input_value(1);
        const auto& weights_shape = weights.get_shape();
        const auto& input_shape = conv_node->get_input_partial_shape(0);
        auto hidden_out = weights_shape[0];
        auto hidden_in = weights_shape[1];

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
                                                   std::vector<int64_t>{(int64_t)hidden_out, (int64_t)hidden_in});
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
        ov::copy_runtime_info(conv_node, {transpose_input, matmul, final_node});
        ov::replace_node(conv_node, final_node);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(conv_pattern, matcher_name);
    this->register_matcher(m, callback);
}
