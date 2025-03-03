// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/istft.hpp"

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_istft(const NodeContext& context) {
    // aten::istft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool
    // center=True, bool normalized=False, bool? onesided=None, int? length=None, bool return_complex=False)
    num_inputs_check(context, 2, 10, true);

    auto input = context.get_input(0);
    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());
    if (complex_type_mark) {
        input = complex_type_mark->input_value(0);
    }

    auto n_fft = context.get_input(1);

    ov::Output<ov::Node> hop_length;
    if (!context.input_is_none(2)) {
        hop_length = context.get_input(2);
    } else {
        // Defualt floor(n_fft / 4)
        const auto four = context.mark_node(std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{}, 4));
        const auto four_cast = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(four, n_fft));
        hop_length = context.mark_node(std::make_shared<ov::op::v1::Divide>(n_fft, four_cast));
    }

    ov::Output<ov::Node> win_length;
    if (!context.input_is_none(3)) {
        win_length = context.get_input(3);
    } else {
        win_length = n_fft;
    }

    ov::Output<ov::Node> window;
    if (!context.input_is_none(4)) {
        window = context.get_input(4);
    } else {
        const auto one = context.mark_node(std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{}, 1));
        const auto one_cast = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(one, input));
        const auto zero = context.mark_node(std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{1}, 0));
        const auto win_length_cast =
            context.mark_node(std::make_shared<ov::op::v0::Convert>(win_length, ov::element::i64));
        const auto win_len_vec = context.mark_node(std::make_shared<ov::op::v0::Unsqueeze>(win_length_cast, zero));
        window = context.mark_node(std::make_shared<ov::op::v3::Broadcast>(one_cast, win_len_vec));
    }

    bool center = true;
    if (!context.input_is_none(5)) {
        center = context.const_input<bool>(5);
    }

    bool normalized = false;
    if (!context.input_is_none(6)) {
        normalized = context.const_input<bool>(6);
    }

    bool onesided = true;
    if (!context.input_is_none(7)) {
        onesided = context.const_input<bool>(7);
    }
    PYTORCH_OP_CONVERSION_CHECK(onesided, "aten::istft conversion is currently supported with onesided=True only.");

    bool return_complex = false;
    if (!context.input_is_none(9)) {
        return_complex = context.const_input<bool>(9);
    }

    // Perform ISTFT
    ov::Output<ov::Node> istft;
    if (context.input_is_none(8)) {
        istft = context.mark_node(std::make_shared<v16::ISTFT>(input, window, n_fft, hop_length, center, normalized));
    } else {
        auto signal_length = context.get_input(8);
        istft = context.mark_node(
            std::make_shared<v16::ISTFT>(input, window, n_fft, hop_length, signal_length, center, normalized));
    }

    if (return_complex) {
        return {context.mark_node(std::make_shared<ComplexTypeMark>(istft, istft.get_element_type()))};
    } else {
        return {istft};
    }
};
}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
